"""Resumable training/evaluation; no experiment starts merely by importing."""
import gc
import hashlib
import importlib.metadata
import io
import json
import math
import random
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from PIL import Image
from torch import nn
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split

from master_workflow import TEXT_MODELS, IMAGE_MODELS, VARIANTS, REVISION, digest, save_json
from baseline_text import metrics


def seed_all(seed):
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    if torch.cuda.is_available(): torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def resolved_revisions(root, spec):
    from huggingface_hub import HfApi
    file = Path(root) / 'model_revisions.json'
    locks = json.loads(file.read_text()) if file.exists() else {}
    result = {}
    for family, names in [('image', IMAGE_MODELS), ('text', TEXT_MODELS)]:
        name = spec.get(family, '')
        if name:
            model_id = names[name]
            if model_id not in locks:
                locks[model_id] = HfApi().model_info(model_id).sha
                assert len(locks[model_id]) == 40
                save_json(file, locks)
            result[family] = {'id': model_id, 'revision': locks[model_id]}
    return result


def tokenizer_and_processor(config):
    from transformers import AutoTokenizer, AutoImageProcessor
    text, image = config['models'].get('text'), config['models'].get('image')
    tokenizer = AutoTokenizer.from_pretrained(text['id'], revision=text['revision']) if text else None
    processor = AutoImageProcessor.from_pretrained(image['id'], revision=image['revision']) if image else None
    return tokenizer, processor


def text_values(frame, variant, tokenizer):
    assert tokenizer.sep_token
    fields = VARIANTS[variant]
    separator = ' </s></s> ' if tokenizer.sep_token == '</s>' else ' ' + tokenizer.sep_token + ' '
    return [separator.join(str(row[c]) for c in fields if str(row[c]).strip())
            for row in frame.to_dict('records')]


def token_audit(frame, tokenizer, root, model_name):
    rows=[]
    for variant in VARIANTS:
        values=text_values(frame,variant,tokenizer)
        lengths=[]
        for start in range(0,len(values),128):
            tokens=tokenizer(values[start:start+128],truncation=False,padding=False,verbose=False)['input_ids']
            lengths.extend(map(len,tokens))
        rows.append({'model':model_name,'variant':variant,'rows':len(lengths),
                     'median_tokens':float(np.median(lengths)),'p95_tokens':float(np.quantile(lengths,.95)),
                     'max_tokens':int(max(lengths)),'fraction_above_256':float(np.mean(np.array(lengths)>256)),
                     'separator':'</s></s>' if tokenizer.sep_token=='</s>' else tokenizer.sep_token})
    result=pd.DataFrame(rows)
    result.to_csv(Path(root)/'eda'/f'tokens_{model_name}.csv',index=False)
    return result


class MemeDataset(Dataset):
    def __init__(self, frame, images, tokenizer, processor, variant, max_length=256):
        self.frame=frame.reset_index(drop=True); self.images=images; self.processor=processor
        self.tokens=(tokenizer(text_values(self.frame,variant,tokenizer),padding='max_length',truncation=True,
                               max_length=max_length,return_tensors='pt') if tokenizer else {})

    def __len__(self): return len(self.frame)

    def __getitem__(self,index):
        row=self.frame.iloc[index]
        item={'image_path':row.image_path, 'labels':torch.tensor([row.y_hate,row.y_appropriate],dtype=torch.long)}
        item.update({key:value[index] for key,value in self.tokens.items() if key in ['input_ids','attention_mask']})
        if self.processor:
            with Image.open(io.BytesIO(self.images[row.image_path])) as image:
                image.load()
                item['pixel_values']=self.processor(images=image.convert('RGB'),return_tensors='pt')['pixel_values'][0]
        return item


class MemeClassifier(nn.Module):
    def __init__(self,config):
        super().__init__()
        from transformers import AutoModel, CLIPVisionModel
        self.config=config; self.text_encoder=None; self.image_encoder=None; dim=0
        for family in ['text','image']:
            if family not in config['models']: continue
            m=config['models'][family]
            cls=CLIPVisionModel if family=='image' and config['image']=='CLIP' else AutoModel
            # Original IndoBERT/IndoBERTweet releases provide .bin; PyTorch >=2.6 loads weights safely.
            assert tuple(map(int,torch.__version__.split('+')[0].split('.')[:2])) >= (2,6)
            encoder=cls.from_pretrained(m['id'],revision=m['revision'])
            setattr(self,family+'_encoder',encoder); dim+=encoder.config.hidden_size
        assert dim>0
        self.dropout=nn.Dropout(config['dropout'])
        needs_projection=config['kind']=='multimodal' or config['task']=='both'
        self.shared=nn.Sequential(nn.Linear(dim,dim//2),nn.Dropout(config['dropout'])) if needs_projection else nn.Identity()
        out_dim=dim//2 if needs_projection else dim
        self.heads=nn.ModuleDict({task:nn.Linear(out_dim,2) for task in
                                (['hate','appropriate'] if config['task']=='both' else [config['task']])})

    def forward(self,batch):
        features=[]
        if self.text_encoder is not None:
            text=self.text_encoder(input_ids=batch['input_ids'],attention_mask=batch['attention_mask'])
            features.append(text.pooler_output)
        if self.image_encoder is not None:
            image=self.image_encoder(pixel_values=batch['pixel_values'])
            features.append(image.last_hidden_state.mean(dim=1) if self.config['image']=='SwinV2' else image.pooler_output)
        feature=torch.cat(features,dim=1) if len(features)>1 else features[0]
        if self.config['kind']=='text': feature=self.dropout(feature)
        elif self.config['kind']=='image' and self.config['task']!='both': feature=self.dropout(feature)
        shared=self.shared(feature)
        return {task:head(shared) for task,head in self.heads.items()}


def total_loss(logits,labels,lambda_hate):
    terms={task:nn.functional.cross_entropy(value,labels[:,0 if task=='hate' else 1]) for task,value in logits.items()}
    if len(terms)==2: return lambda_hate*terms['hate']+(1-lambda_hate)*terms['appropriate']
    return next(iter(terms.values()))


def evaluate(model,loader,device):
    model.eval(); records=[]
    with torch.no_grad():
        for batch in loader:
            labels=batch['labels']; names=batch['image_path']
            inputs={k:v.to(device) for k,v in batch.items() if k not in ['image_path','labels']}
            logits=model(inputs)
            for task,values in logits.items():
                prob=values.softmax(-1).cpu().numpy(); pred=prob.argmax(-1)
                for i,name in enumerate(names):
                    records.append({'image_path':name,'task':task,'truth':int(labels[i,0 if task=='hate' else 1]),
                                    'prediction':int(pred[i]),'probability_positive':float(prob[i,1])})
    frame=pd.DataFrame(records)
    scores={task:metrics(group.truth.to_numpy(),group.prediction.to_numpy()) for task,group in frame.groupby('task')}
    if set(scores)=={'hate','appropriate'}:
        h=frame[frame.task=='hate'].set_index('image_path'); a=frame[frame.task=='appropriate'].set_index('image_path')
        constrained=np.maximum(a.prediction.to_numpy(),h.loc[a.index,'prediction'].to_numpy())
        scores['appropriate_constrained']=metrics(a.truth.to_numpy(),constrained)
        frame['prediction_constrained']=frame.prediction
        frame.loc[frame.task=='appropriate','prediction_constrained']=constrained
    return scores,frame


def capped(frame,count,seed):
    if count is None or len(frame)<=count: return frame.reset_index(drop=True)
    assert count>=2*frame.joint_label.nunique()
    _,selected=train_test_split(frame,test_size=count,stratify=frame.joint_label,random_state=seed)
    return selected.reset_index(drop=True)


def make_config(root,spec,scope='pilot',lambda_hate=.5):
    assert scope in ['pilot','full'] and spec['rq'] in ['RQ2','RQ4']
    assert 0<=lambda_hate<=1
    versions={p:importlib.metadata.version(p) for p in ['torch','transformers','numpy','pandas','scikit-learn']}
    return {**spec,'scope':scope,'seed':42,'epochs':2 if scope=='pilot' else 5,'batch_size':2,'gradient_accumulation':4,
            'learning_rate':2e-5,'weight_decay':.01,'dropout':.3,'max_length':256,'patience':2,'lambda_hate':lambda_hate,
            'max_train':128 if scope=='pilot' else None,'max_validation':64 if scope=='pilot' else None,
            'precision':'float32','optimizer':'AdamW','schedule':'linear, warmup=0','label_map':{'hate':{'not hate':0,'hate':1},
            'appropriate':{'appropriate':0,'inappropriate':1}},'models':resolved_revisions(root,spec),
            'data_revision':REVISION,'split_sha256':digest(Path(root)/'splits/train_validation.csv'),
            'script_sha256':digest(__file__),'support_sha256':{name:digest(Path(__file__).with_name(name))
                for name in ['master_workflow.py','baseline_text.py']},
            'versions':versions,'architecture':'author-code-style pooler/mean; linear-dropout projection for multimodal or dual-head',
            'selection_metric':'mean(hate macro-F1, constrained inappropriate macro-F1)' if spec['task']=='both' else 'macro-F1',
            'comparison_policy':'RQ4 appropriate uses constrained predictions; raw scores retained',
            'deviations':['validation derived from official train; selection by validation macro-F1',
                          'physical batch 2 x accumulation 4 approximates source batch 8; FP32',
                          'early stopping patience 2, warmup 0; clarify with authors',
                          'multitask lambda is an explicit candidate; validation tuning must be documented',
                          'model revisions pinned on first use; original author training revisions unknown',
                          'pooler_output follows source; article pooled-CLS wording requires clarification',
                          'XLM-R double </s></s> follows RQ2 article input diagram; RQ4 prose/source differ',
                          'unimodal dual-head projection follows author code; article prose omits its dimension',
                          'ViT checkpoint follows article (224-in21k), differs from active source 384',
                          'appropriate positive label is inappropriate; macro metrics unchanged by remapping']}


def run_training(root,config,fit,val,images):
    from transformers import get_linear_schedule_with_warmup
    root=Path(root); seed_all(config['seed'])
    assert config['split_sha256']==digest(root/'splits/train_validation.csv')
    payload=json.dumps(config,sort_keys=True,separators=(',',':'))
    run_id=config['experiment_id']+'__'+hashlib.sha256(payload.encode()).hexdigest()[:12]
    run=root/'runs'/run_id; run.mkdir(parents=True,exist_ok=True)
    if (run/'result.json').exists():
        print('Hasil tersimpan dimuat:',run_id)
        return json.loads((run/'result.json').read_text()),run
    save_json(run/'config.json',config)
    train_frame=capped(fit,config['max_train'],config['seed']); val_frame=capped(val,config['max_validation'],config['seed'])
    train_frame[['image_path']].to_csv(run/'fit_ids.csv',index=False); val_frame[['image_path']].to_csv(run/'validation_ids.csv',index=False)
    assert not set(train_frame.image_path)&set(val_frame.image_path)
    tokenizer,processor=tokenizer_and_processor(config)
    train_data=MemeDataset(train_frame,images,tokenizer,processor,config['variant'],config['max_length'])
    val_data=MemeDataset(val_frame,images,tokenizer,processor,config['variant'],config['max_length'])
    val_loader=DataLoader(val_data,batch_size=config['batch_size'],shuffle=False,num_workers=0)
    device=torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model=MemeClassifier(config).to(device)
    optimizer=torch.optim.AdamW(model.parameters(),lr=config['learning_rate'],weight_decay=config['weight_decay'])
    steps_per_epoch=math.ceil(math.ceil(len(train_data)/config['batch_size'])/config['gradient_accumulation'])
    scheduler=get_linear_schedule_with_warmup(optimizer,0,steps_per_epoch*config['epochs'])
    start_epoch=0; best=-1.; bad_epochs=0; history=[]
    if (run/'last.pt').exists():
        checkpoint=torch.load(run/'last.pt',map_location=device,weights_only=True)
        assert checkpoint['config_json']==payload
        model.load_state_dict(checkpoint['model']); optimizer.load_state_dict(checkpoint['optimizer']); scheduler.load_state_dict(checkpoint['scheduler'])
        start_epoch=checkpoint['epoch']+1; best=checkpoint['best']; bad_epochs=checkpoint['bad_epochs']; history=checkpoint['history']
        print('Melanjutkan dari epoch selesai:',start_epoch)
    for epoch in range(start_epoch,config['epochs']):
        if bad_epochs>=config['patience']: break
        seed_all(config['seed']+epoch)
        generator=torch.Generator().manual_seed(config['seed']+epoch)
        loader=DataLoader(train_data,batch_size=config['batch_size'],shuffle=True,generator=generator,num_workers=0)
        model.train(); optimizer.zero_grad(set_to_none=True); losses=[]
        for step,batch in enumerate(loader):
            inputs={k:v.to(device) for k,v in batch.items() if k not in ['image_path','labels']}
            loss=total_loss(model(inputs),batch['labels'].to(device),config['lambda_hate'])
            assert torch.isfinite(loss).item(), 'Loss tidak terhingga; hentikan run.'
            count=min(config['gradient_accumulation'],len(loader)-(step//config['gradient_accumulation'])*config['gradient_accumulation'])
            (loss/count).backward(); losses.append(float(loss.detach().cpu()))
            if (step+1)%config['gradient_accumulation']==0 or step+1==len(loader):
                torch.nn.utils.clip_grad_norm_(model.parameters(),1.0,error_if_nonfinite=True)
                optimizer.step(); scheduler.step(); optimizer.zero_grad(set_to_none=True)
        scores,predictions=evaluate(model,val_loader,device)
        primary=float(np.mean([scores['appropriate_constrained' if t=='appropriate' and len(model.heads)==2 else t]['macro_f1']
                               for t in model.heads]))
        history.append({'epoch':epoch+1,'train_loss':float(np.mean(losses)),'selection_macro_f1':primary,
                        **{task+'_macro_f1':s['macro_f1'] for task,s in scores.items()}})
        improved=primary>best
        if improved:
            best=primary; bad_epochs=0
            temporary=run/'best.pt.tmp'; torch.save({'model':model.state_dict(),'config_json':payload,'epoch':epoch+1},temporary); temporary.replace(run/'best.pt')
            predictions.to_csv(run/'validation_predictions.csv',index=False)
            save_json(run/'validation_metrics.json',scores)
        else: bad_epochs+=1
        pd.DataFrame(history).to_csv(run/'history.csv',index=False)
        temporary=run/'last.pt.tmp'
        torch.save({'model':model.state_dict(),'optimizer':optimizer.state_dict(),'scheduler':scheduler.state_dict(),
                    'epoch':epoch,'best':best,'bad_epochs':bad_epochs,'history':history,'config_json':payload},temporary)
        temporary.replace(run/'last.pt')
        print(f"Epoch {epoch+1}: loss={history[-1]['train_loss']:.4f}, validation macro-F1={primary:.4f}; checkpoint tersimpan.",flush=True)
    result={'config':config,'run_id':run_id,'evaluation':'validation','best_selection_macro_f1':best,
            'metrics':json.loads((run/'validation_metrics.json').read_text()),'epochs_completed':len(history),'test_evaluated':False,
            'resume_boundary':'last completed epoch; interrupted epoch is replayed','official_paper_condition_completed':False}
    save_json(run/'result.json',result)
    del optimizer,model,train_data,val_data; gc.collect()
    if torch.cuda.is_available(): torch.cuda.empty_cache()
    return result,run


def lock_test_protocol(root,run_ids):
    assert run_ids and len(set(run_ids))==len(run_ids)
    records=[]
    for run_id in run_ids:
        run=Path(root)/'runs'/run_id; result=json.loads((run/'result.json').read_text()); c=result['config']
        assert c['scope']=='full' and c['max_train'] is None and c['max_validation'] is None, 'Pilot tidak boleh menjadi hasil paper.'
        records.append({'run_id':run_id,'config_sha256':digest(run/'config.json'),'checkpoint_sha256':digest(run/'best.pt')})
    value={'purpose':'Frozen selection before official test evaluation','runs':records,
           'split_sha256':digest(Path(root)/'splits/train_validation.csv')}
    path=Path(root)/'reports/test_protocol_lock.json'
    if path.exists(): assert json.loads(path.read_text())==value, 'Protokol test sudah terkunci; jangan tuning ulang memakai test.'
    else: save_json(path,value)
    return value


def evaluate_official_test(root,run_id,test,images):
    root=Path(root); lock=json.loads((root/'reports/test_protocol_lock.json').read_text())
    record=next((r for r in lock['runs'] if r['run_id']==run_id),None); assert record, 'Run belum masuk protokol terkunci.'
    run=root/'runs'/run_id
    assert digest(run/'config.json')==record['config_sha256'] and digest(run/'best.pt')==record['checkpoint_sha256']
    if (run/'test_metrics.json').exists(): return json.loads((run/'test_metrics.json').read_text())
    assert len(test)==1005 and test.split.eq('test').all() and test.image_path.is_unique
    config=json.loads((run/'config.json').read_text()); tokenizer,processor=tokenizer_and_processor(config)
    data=MemeDataset(test,images,tokenizer,processor,config['variant'],config['max_length'])
    device=torch.device('cuda' if torch.cuda.is_available() else 'cpu'); model=MemeClassifier(config).to(device)
    model.load_state_dict(torch.load(run/'best.pt',map_location=device,weights_only=True)['model'])
    scores,predictions=evaluate(model,DataLoader(data,batch_size=config['batch_size'],shuffle=False,num_workers=0),device)
    predictions.to_csv(run/'test_predictions.csv',index=False); save_json(run/'test_metrics.json',scores)
    result=json.loads((run/'result.json').read_text())
    result['test_evaluated']=True
    result['official_test_evaluation_completed']=True
    save_json(run/'result.json',result)
    del model,data; gc.collect()
    if torch.cuda.is_available(): torch.cuda.empty_cache()
    return scores


def comparison(root,references):
    rows=[]
    for file in Path(root).glob('runs/*/result.json'):
        result=json.loads(file.read_text()); test=file.parent/'test_metrics.json'
        scores=json.loads(test.read_text()) if test.exists() else result['metrics']
        for task,value in scores.items():
            if task not in ['hate','appropriate']: continue
            reported=scores.get('appropriate_constrained',value) if task=='appropriate' and result['config']['task']=='both' else value
            rows.append({'experiment_id':result['config']['experiment_id'],'run_id':file.parent.name,'task':task,
                         'scope':result['config']['scope'],'evaluation':'test' if test.exists() else 'validation',
                         'prediction_policy':'constrained' if reported is not value else 'raw',
                         'our_raw_macro_f1':value['macro_f1'],
                         'our_macro_precision':reported['macro_precision'],'our_macro_recall':reported['macro_recall'],
                         'our_macro_f1':reported['macro_f1'],'our_accuracy':reported['accuracy']})
    for file in Path(root).glob('reports/rq3/*/metrics.json'):
        result=json.loads(file.read_text()); value=result['scores']
        matches=references[references.experiment_id==file.parent.name]
        assert len(matches)==1, 'Kondisi RQ3 tidak ada dalam referensi.'
        rows.append({'experiment_id':file.parent.name,'run_id':file.parent.name,'task':matches.iloc[0].task,
                     'scope':'zero-shot','evaluation':'test','prediction_policy':'raw','our_raw_macro_f1':value['macro_f1'],
                     **{'our_'+key:value[key] for key in ['macro_precision','macro_recall','macro_f1','accuracy']}})
    columns=['experiment_id','run_id','task','scope','evaluation','prediction_policy','our_raw_macro_f1',
             'our_macro_precision','our_macro_recall','our_macro_f1','our_accuracy']
    table=pd.DataFrame(rows,columns=columns).merge(references,on=['experiment_id','task'],how='left',validate='many_to_one')
    table['delta_macro_f1']=np.where(table.scope.isin(['full','zero-shot'])&(table.evaluation=='test'),
                                   table.our_macro_f1-table.paper_macro_f1,np.nan)
    table.to_csv(Path(root)/'reports/paper_comparison.csv',index=False)
    return table


def import_rq3(root,csv_path,test,plan,prompt_path):
    """Evaluate externally generated, audited zero-shot predictions; never invoke an API implicitly."""
    frame=pd.read_csv(csv_path,keep_default_na=False,dtype={'raw_response':str,'model_version':str,'prompt_sha256':str,'image_path':str})
    required={'image_path','experiment_id','prediction','raw_response','model_version','prompt_sha256'}
    assert len(test)==1005 and test.split.eq('test').all() and test.image_path.is_unique
    assert required<=set(frame) and frame.image_path.is_unique and set(frame.image_path)==set(test.image_path)
    assert len(frame)==1005 and frame.experiment_id.nunique()==1
    spec=plan.set_index('experiment_id').loc[frame.experiment_id.iloc[0]].to_dict(); assert spec['rq']=='RQ3'
    assert frame.prompt_sha256.eq(digest(prompt_path)).all()
    assert frame.raw_response.str.strip().ne('').all() and frame.model_version.str.strip().ne('').all()
    assert frame.model_version.nunique()==1, 'Satu kondisi wajib memakai satu versi model.'
    assert set(frame.prediction)<= {0,1}, 'Prediction wajib 0/1 sesuai label map notebook.'
    truth=test.set_index('image_path')['y_'+spec['task']].loc[frame.image_path].to_numpy()
    scores=metrics(truth,frame.prediction.to_numpy())
    folder=Path(root)/'reports/rq3'/frame.experiment_id.iloc[0]; folder.mkdir(parents=True,exist_ok=True)
    if (folder/'metrics.json').exists():
        previous=json.loads((folder/'metrics.json').read_text())
        assert previous['source_sha256']==digest(csv_path) and previous['prompt_sha256']==digest(prompt_path), 'Hasil RQ3 sudah terkunci; jangan mengganti prediksi setelah melihat test.'
        return previous['scores']
    frame.to_csv(folder/'predictions.csv',index=False)
    (folder/'prompt.txt').write_bytes(Path(prompt_path).read_bytes())
    save_json(folder/'metrics.json',{'scores':scores,'source_sha256':digest(csv_path),'prompt_sha256':digest(prompt_path),
                                   'evaluation':'official test','model_versions':frame.model_version.unique().tolist(),
                                   'inference_executed_by_this_module':False})
    return scores
