import pandas as pd
import pytest
import torch

from cp2.nlp_data import clean_field, compose_text, make_content_mask, MemeTextDataset
from cp2.nlp_evaluation import task_scores, evaluate_records, rescue_hurt
from cp2.nlp_runner import model_loss, selection_value


class DummyTokenizer:
    sep_token = "[SEP]"
    all_special_ids = [0, 101, 102]
    def __call__(self, texts, **kwargs):
        if isinstance(texts, str):
            texts = [texts]
        data = []
        for text in texts:
            ids = [101] + [102 if word == "[SEP]" else (11 + len(word)) for word in text.split()] + [102]
            max_length = kwargs.get("max_length", 256)
            ids = ids[:max_length]
            if kwargs.get("padding") == "max_length":
                ids = ids + [0] * (max_length - len(ids))
            data.append(ids)
        return {"input_ids": torch.tensor(data),
                "attention_mask": torch.tensor([[int(x != 0) for x in v] for v in data])}


def preds():
    return pd.DataFrame([
        {"image_path": image_id, "task": task, "truth": truth, "prediction_raw": pred, "probability_positive": 0.8}
        for task, truths, predictions in [
            ("hate", [0, 0, 1, 1], [0, 1, 1, 1]),
            ("appropriate", [0, 1, 1, 1], [0, 0, 1, 1]),
        ]
        for image_id, truth, pred in zip(["a", "b", "c", "d"], truths, predictions)
    ])


def test_clean_null_and_separator():
    assert clean_field(None) == ""
    assert clean_field(float("nan")) == ""
    assert compose_text({"caption": " Halo ", "ocr": "teks"}, "[SEP]") == "Halo [SEP] teks"
    assert compose_text({"caption": "nan", "ocr": "teks"}, "[SEP]") == "teks"
    assert compose_text({"caption": "halo", "ocr": "teks"}, "[SEP]", "no_caption") == "teks"
    assert compose_text({"caption": "halo", "ocr": "teks"}, "[SEP]", "no_ocr") == "halo"
    with pytest.raises(ValueError):
        compose_text({"image_path": "X", "caption": None, "ocr": None}, "[SEP]")


def test_content_mask_excludes_inner_sep_and_pad():
    ids = torch.tensor([[101, 20, 102, 30, 102, 0], [101, 102, 0, 0, 0, 0]])
    attention = torch.tensor([[1, 1, 1, 1, 1, 0], [1, 1, 0, 0, 0, 0]])
    actual = make_content_mask(ids, attention, [0, 101, 102])
    assert actual.dtype == torch.bool
    assert actual.tolist() == [[False, True, False, True, False, False], [False] * 6]


def test_dataset_contract():
    frame = pd.DataFrame([
        {"image_path": "a", "caption": "abc", "ocr": "xyz", "y_hate": 0, "y_appropriate": 1},
        {"image_path": "b", "caption": "foo", "ocr": "bar", "y_hate": 1, "y_appropriate": 1},
    ])
    dataset = MemeTextDataset(frame, DummyTokenizer(), max_length=8)
    sample = dataset[0]
    assert sample["input_ids"].shape == (8,)
    assert sample["attention_mask"].shape == (8,)
    assert sample["content_mask"].shape == (8,)
    assert sample["labels"].tolist() == [0, 1]
    assert sample["content_mask"].sum() == 2


def test_metrics_reference_fixture():
    result = task_scores([0, 0, 1, 1], [0, 1, 1, 1])
    assert result["confusion_matrix_0_1"] == [[1, 1], [0, 2]]
    assert result["accuracy"] == pytest.approx(0.75)
    assert result["macro_f1"] == pytest.approx(11 / 15)


def test_evaluation_raw_constrained_and_pairing():
    p = preds()
    # Reorder appropriate rows to force ID-based join, not accidental row order.
    p = pd.concat([p[p.task == "hate"], p[p.task == "appropriate"].iloc[::-1]], ignore_index=True)
    scores, out = evaluate_records(p)
    assert scores["raw_constraint_violations"] == 1
    assert scores["constrained_constraint_violations"] == 0
    assert selection_value(scores) == pytest.approx((scores["hate"]["macro_f1"] + scores["appropriate"]["macro_f1"]) / 2)
    assert set(out.columns) >= {"prediction_raw", "prediction_constrained"}
    assert out.loc[(out.image_path == "b") & (out.task == "appropriate"), "prediction_constrained"].iloc[0] == 1


def test_duplicate_and_unpaired_rejected():
    p = preds()
    with pytest.raises(ValueError):
        evaluate_records(pd.concat([p, p.iloc[[0]]]))
    with pytest.raises(ValueError):
        evaluate_records(p.drop(index=p.index[-1]))


def test_rescue_hurt_by_id_and_task():
    reference = preds()
    new = reference.copy()
    new.loc[(new.image_path == "b") & (new.task == "hate"), "prediction_raw"] = 0
    new = new.iloc[::-1]
    report = rescue_hurt(reference, new)
    assert report["hate"] == {"rescue": 1, "hurt": 0}
    assert report["appropriate"] == {"rescue": 0, "hurt": 0}


def test_loss_finite_grad():
    logit_h = torch.randn(2, 2, requires_grad=True)
    logit_a = torch.randn(2, 2, requires_grad=True)
    labels = torch.tensor([[0, 0], [1, 1]])
    loss = model_loss({"hate": logit_h, "appropriate": logit_a}, labels)
    assert loss.isfinite().item()
    loss.backward()
    assert logit_h.grad is not None and logit_a.grad is not None

@pytest.mark.parametrize("mode", [
    "xa_concat", "xa_residual", "xa_shared_scalar", "xa_task_scalar",
    "xa_shared_vector", "xa_task_vector", "xa_shared_scalar_dropout", "xa_task_scalar_dropout",
])
def test_reference_fusion_output_shapes_and_zero_visual(mode):
    from cp2.fusion_reference import CrossAttentionReference
    fuser = CrossAttentionReference(mode, dim=16, heads=4, dropout=0.0)
    fuser.eval()
    t = torch.randn(2, 16)
    tokens = torch.randn(2, 6, 16)
    patches = torch.randn(2, 49, 16)
    mask = torch.tensor([[True] * 4 + [False] * 2, [False] * 6])
    out = fuser(t, tokens, patches, mask)
    off = fuser(t, tokens, patches, mask, visual_off=True)
    assert set(out) == {"hate", "appropriate"}
    assert all(v.shape == (2, 2) and torch.isfinite(v).all() for v in out.values())
    assert all(torch.isfinite(v).all() for v in off.values())
    pooled = fuser.masked_mean(torch.ones(2, 6, 16), mask)
    assert pooled[1].eq(0).all()


def test_shared_scalar_and_task_specific_gate_have_distinct_shapes():
    from cp2.fusion_reference import CrossAttentionReference
    s = CrossAttentionReference("xa_shared_scalar", dim=16, heads=4)
    t = CrossAttentionReference("xa_task_scalar", dim=16, heads=4)
    v = CrossAttentionReference("xa_task_vector", dim=16, heads=4)
    assert s.shared_gate.out_features == 1
    assert t.task_gates["hate"].out_features == 1
    assert t.task_gates["hate"] is not t.task_gates["appropriate"]
    assert v.task_gates["hate"].out_features == 16


def test_training_only_visual_dropout_has_no_eval_randomness():
    from cp2.fusion_reference import CrossAttentionReference
    model = CrossAttentionReference("xa_shared_scalar_dropout", dim=16, heads=4, dropout=0.0)
    t = torch.randn(2, 16)
    tokens = torch.randn(2, 6, 16)
    patches = torch.randn(2, 49, 16)
    mask = torch.ones(2, 6, dtype=torch.bool)
    model.eval()
    a = model(t, tokens, patches, mask)
    b = model(t, tokens, patches, mask)
    assert all(torch.equal(a[k], b[k]) for k in a)


def test_partition_load_and_missing_id_rejection(tmp_path):
    from cp2.nlp_data import get_partitions
    meta = pd.DataFrame([
        {"image_path": "a", "caption": "a", "ocr": "a", "y_hate": 0, "y_appropriate": 0, "split": "train"},
        {"image_path": "b", "caption": "b", "ocr": "b", "y_hate": 1, "y_appropriate": 1, "split": "train"},
        {"image_path": "c", "caption": "c", "ocr": "c", "y_hate": 0, "y_appropriate": 1, "split": "test"},
    ])
    manifest = pd.DataFrame([{"image_path": "a", "partition": "train"}, {"image_path": "b", "partition": "validation"}])
    meta_path = tmp_path / "data/processed/metadata.csv"
    manifest_path = tmp_path / "splits/train_validation.csv"
    meta_path.parent.mkdir(parents=True)
    manifest_path.parent.mkdir(parents=True)
    meta.to_csv(meta_path, index=False)
    manifest.to_csv(manifest_path, index=False)
    frames, digest = get_partitions(tmp_path, enforce_expected=False)
    assert len(frames["fit"]) == 1 and len(frames["validation"]) == 1 and len(frames["test"]) == 1
    assert len(digest) == 64
    manifest.loc[1, "image_path"] = "missing"
    manifest.to_csv(manifest_path, index=False)
    with pytest.raises(ValueError):
        get_partitions(tmp_path, enforce_expected=False)


def test_visual_dropout_one_is_visual_off_in_training():
    from cp2.fusion_reference import CrossAttentionReference
    model = CrossAttentionReference("xa_shared_scalar_dropout", dim=16, heads=4, dropout=0, visual_dropout_p=1.0)
    model.train()
    t = torch.randn(2, 16)
    tokens = torch.randn(2, 4, 16)
    patches = torch.randn(2, 49, 16)
    mask = torch.ones(2, 4, dtype=torch.bool)
    dropped = model(t, tokens, patches, mask)
    explicit_off = model(t, tokens, patches, mask, visual_off=True)
    for task in ("hate", "appropriate"):
        assert torch.equal(dropped[task], explicit_off[task])


def test_reference_fusion_gradient_reaches_attention_and_task_gates():
    from cp2.fusion_reference import CrossAttentionReference
    model = CrossAttentionReference("xa_task_scalar", dim=16, heads=4, dropout=0)
    t = torch.randn(2, 16, requires_grad=True)
    tokens = torch.randn(2, 5, 16, requires_grad=True)
    patches = torch.randn(2, 49, 16, requires_grad=True)
    mask = torch.ones(2, 5, dtype=torch.bool)
    output = model(t, tokens, patches, mask)
    sum(tensor.square().sum() for tensor in output.values()).backward()
    assert model.attention.in_proj_weight.grad is not None
    assert model.task_gates["hate"].weight.grad is not None
    assert model.task_gates["appropriate"].weight.grad is not None
    assert patches.grad is not None and torch.isfinite(patches.grad).all()


def test_cv_parameter_counts_reference():
    from cp2.fusion_reference import CrossAttentionReference
    m1 = CrossAttentionReference("xa_residual")
    m2 = CrossAttentionReference("xa_shared_scalar")
    m3 = CrossAttentionReference("xa_shared_scalar_dropout")
    counts = [sum(p.numel() for p in model.parameters()) for model in (m1, m2, m3)]
    assert counts == [2365444, 2366981, 2366981]


def test_invalid_ground_truth_constraint_rejected():
    p = preds()
    p.loc[(p.image_path == "c") & (p.task == "appropriate"), "truth"] = 0
    with pytest.raises(ValueError, match="Ground-truth"):
        evaluate_records(p)


def test_pilot_full_data_requires_one_epoch(monkeypatch, tmp_path):
    from cp2.nlp_runner import cmd_train_b0
    from argparse import Namespace
    args = Namespace(full=False, pilot_full_data=True, epochs=2, max_train=32,
                     max_validation=16)
    with pytest.raises(ValueError, match="tepat 1 epoch"):
        cmd_train_b0(args)


def test_b0_runner_one_epoch_synthetic_without_download(monkeypatch, tmp_path):
    import sys
    from types import SimpleNamespace
    from torch import nn
    from cp2 import nlp_runner

    class TinyTextClassifier(nn.Module):
        def __init__(self):
            super().__init__()
            self.embedding = nn.Embedding(200, 8)
            self.heads = nn.ModuleDict({task: nn.Linear(8, 2) for task in ('hate', 'appropriate')})

        def forward(self, batch):
            pooled = self.embedding(batch['input_ids'][:, :3]).mean(dim=1)
            return {task: head(pooled) for task, head in self.heads.items()}

    rows = [
        dict(image_path=f'img_{i}', caption='abc', ocr='xyz', y_hate=i % 2, y_appropriate=i % 2)
        for i in range(6)
    ]
    fit, validation = pd.DataFrame(rows[:4]), pd.DataFrame(rows[4:])
    monkeypatch.setattr(nlp_runner, 'get_partitions', lambda workspace: ({'fit': fit, 'validation': validation}, 'a' * 64))
    monkeypatch.setattr(nlp_runner, 'make_provenance', lambda manifest_sha: {'manifest_sha256': manifest_sha})
    monkeypatch.setattr(nlp_runner, 'tokenizer', lambda: DummyTokenizer())
    monkeypatch.setattr(nlp_runner, 'TextOnlyB0', TinyTextClassifier)
    fake_transformers = SimpleNamespace(
        get_linear_schedule_with_warmup=lambda optim, num_warmup_steps, num_training_steps: 
            torch.optim.lr_scheduler.LambdaLR(optim, lr_lambda=lambda step: max(0., 1. - step / num_training_steps))
    )
    monkeypatch.setitem(sys.modules, 'transformers', fake_transformers)
    args = SimpleNamespace(full=False, pilot_full_data=False, max_train=4, max_validation=2,
                           epochs=1, require_gpu=False, seed=42, workspace=tmp_path,
                           output=tmp_path / 'private', resume=False)
    nlp_runner.cmd_train_b0(args)
    run = args.output / 'B0_smoke_seed42'
    assert (run / 'best.pt').exists() and (run / 'last.pt').exists()
    assert (run / 'validation_metrics.json').exists()
    assert (run / 'validation_predictions_PRIVATE.csv').exists()
    result = __import__('json').loads((run / 'result.json').read_text())
    assert result['epochs_completed'] == 1 and result['test_evaluated'] is False
    with pytest.raises(FileExistsError):
        nlp_runner.cmd_train_b0(args)
