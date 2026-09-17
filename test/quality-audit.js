'use strict';

const assert = require('node:assert/strict');
const fs = require('node:fs');
const path = require('node:path');
const vm = require('node:vm');

const html = fs.readFileSync(path.join(__dirname, '..', 'index.html'), 'utf8');

function extractFunction(name){
  const start = html.indexOf(`function ${name}(`);
  assert.notEqual(start, -1, `${name} should exist`);
  const brace = html.indexOf('{', start);
  let depth = 0;
  for(let i=brace; i<html.length; i++){
    if(html[i]==='{') depth++;
    if(html[i]==='}'){
      depth--;
      if(depth===0) return html.slice(start, i+1);
    }
  }
  throw new Error(`could not extract ${name}`);
}

function evaluateFunction(name){
  return vm.runInNewContext(`(${extractFunction(name)})`);
}

const segments = evaluateFunction('illustrativeTokenSegments');
assert.deepEqual(Array.from(segments('transformer')), ['tran', '##sfo', '##rme', '##r']);
assert.deepEqual(Array.from(segments('hat')), ['hat']);
assert.match(html, /function tokenize\(text\)[\s\S]*illustrativeTokenSegments\(w\)/);
assert.doesNotMatch(html, /function tokenize\(text\)[\s\S]*return toks\.slice\(0,8\)/);

const mlaCacheInfo = evaluateFunction('mlaCacheInfo');
const mla = mlaCacheInfo({latent:512, ropeDim:64});
assert.equal(mla.total, 576);
assert.match(mla.label, /512-dim latent plus 64-dim rotary key/);
assert.match(mla.math, /k_R/);

const routerGateInfo = evaluateFunction('routerGateInfo');
assert.deepEqual(JSON.parse(JSON.stringify(routerGateInfo({moeScore:'sigmoid', topk:8}))), {
  operation:'sigmoid',
  sub:'independent scores',
  tour:'sigmoid-score each expert, keep 8, then renormalize the survivors'
});
assert.equal(routerGateInfo({moeScore:'softmax', topk:2}).operation, 'softmax');

for(const phrase of [
  'does not load model weights or a model tokenizer',
  'illustrative source values',
  'Sources for the displayed configuration',
  'first_k_dense_replace',
  'qk_rope_head_dim',
  'scoring_func'
]) assert.match(html, new RegExp(phrase.replace(/[.*+?^${}()|[\]\\]/g, '\\$&')));

for(const stale of [
  'over 95% of an LLM',
  'same direction, unit scale',
  'The MLP, where facts live',
  'Only the tiny latent is cached',
  'softmax the survivors'
]) assert.doesNotMatch(html, new RegExp(stale.replace(/[.*+?^${}()|[\]\\]/g, '\\$&')));

console.log('quality audit passed');
