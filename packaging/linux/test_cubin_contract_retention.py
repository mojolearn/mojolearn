"""Successful compiler receipt retention, exact reuse and tamper refusal."""
import hashlib
import json
from pathlib import Path
import shlex

import pytest
import cubin_contract as cubin


@pytest.fixture
def compiled(tmp_path):
    count=tmp_path/'executions'
    ptxas=tmp_path/'ptxas'; fatbinary=tmp_path/'fatbinary'
    for tool,source,target in ((ptxas,'m.ptx','m.cubin'),(fatbinary,'m.cubin','m.fatbin')):
        tool.write_text('#!/bin/sh\nif [ "$1" = "--version" ]; then echo "V12.5.82"; exit 0; fi\necho run >> '+shlex.quote(str(count))+'\ncp '+source+' '+target+'\necho compiler-log\n')
        tool.chmod(0o755)
    raw=b'.version 8.5\n.target sm_89\n'
    directory=tmp_path/'retained'; tools=cubin.tool_identities(str(ptxas),str(fatbinary))
    result=cubin.compile_ptx(raw,'sm_89',str(ptxas),str(fatbinary),retention_dir=directory)
    return raw,directory,tools,ptxas,fatbinary,count,result


def test_exact_cached_execution_preserves_all_input_output_and_logs(compiled):
    raw,directory,tools,ptxas,fatbinary,count,result=compiled
    assert count.read_text().splitlines()==['run','run']
    assert cubin.retained_module(directory,raw,'sm_89',tools)==result==raw
    assert cubin.compile_ptx(raw,'sm_89',str(ptxas),str(fatbinary),retention_dir=directory)==result
    assert count.read_text().splitlines()==['run','run']


@pytest.mark.parametrize('name',['m.ptx','m.cubin','m.fatbin','ptxas.stdout','ptxas.stderr','fatbinary.stdout','fatbinary.stderr'])
def test_changed_retained_bytes_refused(compiled,name):
    raw,directory,tools,*_=compiled
    (directory/hashlib.sha256(raw).hexdigest()/name).write_bytes(b'changed')
    with pytest.raises(ValueError,match='bytes differ'):cubin.retained_module(directory,raw,'sm_89',tools)


@pytest.mark.parametrize('field,value',[('arch','sm_86'),('exit_codes',[0,1]),('commands',[]),('tools',{}),('schema','unknown')])
def test_receipt_cannot_replace_execution_facts(compiled,field,value):
    raw,directory,tools,*_=compiled;p=directory/hashlib.sha256(raw).hexdigest()/'receipt.json';r=json.loads(p.read_text());r[field]=value;p.write_text(json.dumps(r))
    with pytest.raises(ValueError,match='recipe/source'):cubin.retained_module(directory,raw,'sm_89',tools)


def test_tool_change_does_not_reuse_or_overwrite_original(compiled):
    raw,directory,tools,ptxas,fatbinary,count,_=compiled
    ptxas.write_text(ptxas.read_text()+'\n# different compiler bytes\n')
    with pytest.raises(ValueError,match='recipe/source'):cubin.compile_ptx(raw,'sm_89',str(ptxas),str(fatbinary),retention_dir=directory)
    assert count.read_text().splitlines()==['run','run']
    assert cubin.retained_module(directory,raw,'sm_89',tools)==raw
