"""One structured equation tree for native Word math and browser MathML."""
from copy import deepcopy
from lxml import etree as ET

MM = 'http://www.w3.org/1998/Math/MathML'
OM = 'http://schemas.openxmlformats.org/officeDocument/2006/math'


def node(tag, *children, text=None):
    e = ET.Element('{'+MM+'}'+tag)
    e.text = text
    for c in children:
        e.append(deepcopy(c))
    return e


def row(*children):
    return node('mrow', *[c if isinstance(c, ET._Element) else node('mi', text=str(c)) for c in children])


def txt(value): return node('mtext', text=value)
def op(value): return node('mo', text=value)
def num(value): return node('mn', text=str(value))
def sub(base, index): return node('msub', row(base), row(index))
def sup(base, power): return node('msup', row(base), row(power))
def frac(top, bottom): return node('mfrac', row(top), row(bottom))
def lower(base, limit): return node('munder', row(base), row(limit))
def array(*lines): return node('mtable', *[node('mtr', node('mtd', line)) for line in lines])
def summation(limit, top, body):
    return row(node('munderover', op('∑'), row(limit), row(top)), body)


def to_omml(e):
    tag = ET.QName(e).localname
    def om(name, *children):
        out = ET.Element('{'+OM+'}'+name)
        for c in children: out.append(c)
        return out
    if tag in ('mi', 'mn', 'mo', 'mtext'):
        t = om('t'); t.text = e.text or ''
        r = om('r', t)
        if tag != 'mi':
            p = om('rPr'); style = om('sty'); style.set('{'+OM+'}val', 'p'); p.append(style); r.insert(0, p)
        return [r]
    if tag == 'mrow' and len(e) == 2 and ET.QName(e[0]).localname == 'munderover':
        bounds = e[0]
        prop = om('naryPr'); char = om('chr'); char.set('{'+OM+'}val', '∑'); prop.append(char)
        loc = om('limLoc'); loc.set('{'+OM+'}val', 'undOvr'); prop.append(loc)
        return [om('nary', prop, om('sub', *to_omml(bounds[1])), om('sup', *to_omml(bounds[2])), om('e', *to_omml(e[1])))]
    if tag == 'mtable': return [om('eqArr', *[om('e', *to_omml(c[0][0])) for c in e])]
    children = [to_omml(c) for c in e]
    if tag in ('math', 'mrow'):
        return [item for group in children for item in group]
    if tag == 'mfrac': return [om('f', om('num', *children[0]), om('den', *children[1]))]
    if tag == 'msub': return [om('sSub', om('e', *children[0]), om('sub', *children[1]))]
    if tag == 'msup': return [om('sSup', om('e', *children[0]), om('sup', *children[1]))]
    if tag == 'munder': return [om('limLow', om('e', *children[0]), om('lim', *children[1]))]
    raise ValueError('Unsupported equation node: '+tag)


def word_math(tree):
    out = ET.Element('{'+OM+'}oMath', nsmap={'m':OM})
    for child in to_omml(tree): out.append(child)
    return out


def browser_math(tree, alt, display=True):
    out = ET.Element('{'+MM+'}math', nsmap={None:MM}); out.append(deepcopy(tree))
    out.set('display', 'block' if display else 'inline'); out.set('aria-label', alt)
    return ET.tostring(out, encoding='unicode')


if __name__ == '__main__':
    sample = row(sub('F', num(1)), op('='), frac(row(num(2), op('×'), 'P', op('×'), 'R'), row('P', op('+'), 'R')))
    native = word_math(sample)
    assert len(native.findall('.//{'+OM+'}f')) == 1
    assert len(native.findall('.//{'+OM+'}sSub')) == 1
    assert 'mfrac' in browser_math(sample, 'F1')
    print('Native equation structure self-check passed')
