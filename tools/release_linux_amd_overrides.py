"""Admit current per-binding AMD groups over proven unchanged merged members.

The existing loader already prefers a member's own file. The effective index
records that ownership explicitly; unused older exports are never represented
as current compiled inputs. No kernel or loader is generated here.
"""
from pathlib import Path
import release_host_composition as hc
import release_retained_compile_evidence as common
import release_reuse
from release_linux_development_evidence import ROUTING, group_facts


def require(ok, why):
    hc.require(ok, 'AMD group overrides: '+why)


def effective_plan(plan, names):
    names=list(names)
    members=[n for rows in plan.values() for n in rows]
    require(len(names)==len(set(names)) and set(names)<=set(members), 'unknown or duplicate override binding')
    result={filename:[n for n in rows if n not in names] for filename,rows in plan.items()}
    require(all(result.values()), 'fully replaced merged groups must be retired explicitly, not retained unused')
    for name in names:
        filename=name+'.so'
        require(filename not in result, 'own filename collides with a merged group')
        result[filename]=[name]
    return result


def environment_policy(env):
    require(not common.derived_build_flags(env, ROUTING)
            and env.get('MOJOLEARN_NUMERIC_MODE')=='identical'
            and env.get('MOJOLEARN_TARGET_COLUMN')=='amd'
            and env.get('MOJOLEARN_GPU_ARCHS')=='gfx942'
            and env.get('MOJOLEARN_LINUX_CPU','x86-64-v3')=='x86-64-v3'
            and env.get('MOJOLEARN_PACKAGE_BYTE_LM')=='1',
            'target, mode or compile flags differ')


def device_groups(rows, root, revision, site, compiler, patchelf, evidence):
    import verify_linux_surface_qualification as surface
    from release_linux_group_composition import PREFIX, DEVICE_RUNPATH, replay
    names=[r['binding'] for r in rows]
    require(len(names)==len(set(names)) and set(names)<=set(surface.expected_bindings('identical',True)),
            'unknown or duplicate individual group')
    tree=common.Tree.at(str(root),revision)
    extensions,origins={},{}
    for spec in rows:
        result_path=evidence.file(spec['result']);result=evidence.json(result_path)
        receipt_path=evidence.file(spec['receipt']);receipt=evidence.json(receipt_path)
        selected=[r for r in result['plan']['groups'] if r['key']==receipt['key']]
        require(len(selected)==1,'ambiguous group execution')
        facts=group_facts(selected[0],result,result_path,receipt_path,tree)
        name=spec['binding']
        require(facts['mode']=='identical' and facts['members']==[name]
                and facts['compiler_sha256']==compiler,'group membership, mode or compiler differs')
        require(facts['toolchain_locked']==release_reuse.toolchain(root,common.LINUX),
                'locked compiler/dependencies differ')
        environment_policy(facts['environment'])
        for field in ('receipt','execution','log','binary'):evidence.file(facts[field]['path'])
        target=Path(site)/'identical'/(name+'.so');key=PREFIX+'identical/'+target.name
        extensions[key]=replay(facts['binary']['path'],target,DEVICE_RUNPATH,patchelf,evidence)
        origins[key]=dict(kind='development-group',key=receipt['key'],binding=name,
                         raw_sha256=facts['binary']['sha256'],sha256=extensions[key],
                         compiler_workers=facts['compiler_workers'],
                         producer_receipt=str(receipt_path),producer_result=str(result_path))
    return extensions,origins
