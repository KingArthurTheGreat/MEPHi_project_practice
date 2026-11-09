const $ = (id) => document.getElementById(id);
let csv = '', run = null, taxonomy = null, selected = null;
const actionNames = {change:'Предложено изменение', review:'Нужен разбор', keep:'Оставить метку'};
const agentNames = {instruction:'Инструкция · Илья', semantic:'Смысл · Артем', consistency:'Согласованность · Роман', arbiter:'Арбитр · Артур', queue:'Очередь · Максим'};
function node(tag, text, cls) {const el=document.createElement(tag); if(text!==undefined)el.textContent=text; if(cls)el.className=cls; return el;}
function labelName(id){return taxonomy?.classes.find(c=>c.id===id)?.name || id || 'Нужен человек';}
function fail(err){$('error').textContent=err.message || String(err);$('error').hidden=false;}
function busy(value, text='Пять агентов проверяют обращения…'){$('run').disabled=value;$('demo').disabled=value;$('csv-file').disabled=value;$('progress').hidden=!value;$('progress').textContent=text;}
async function request(url, payload){const r=await fetch(url,payload===undefined?{}:{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify(payload)});const data=await r.json();if(!r.ok)throw Error(data.error || 'Ошибка запроса');return data;}
async function demo(){csv=await (await fetch('/api/demo')).text();$('file-name').textContent='Учебный набор · 120 синтетических обращений';}
function visibleRows(){const mode=$('filter').value; const reviewed=new Set(run.feedback.map(f=>f.record_id));if(mode==='queue'){const map=new Map(run.findings.map(f=>[f.id,f]));return run.queue.selected_ids.map(id=>map.get(id));}return run.findings.filter(f=>mode==='all'||(mode==='human'?reviewed.has(f.id):f.decision.action===mode));}
function render(){
 $('results').hidden=false;$('empty').hidden=true;$('total').textContent=run.findings.length;$('candidates').textContent=run.queue.candidate_count;$('queued').textContent=run.queue.selected_ids.length;$('reviewed').textContent=new Set(run.feedback.map(f=>f.record_id)).size;
 $('queue-reason').textContent=run.queue.reason;$('disclosure').textContent=run.mode==='demo'?'Демо: эвристики, без вызовов LLM. Синтетические данные предназначены для проверки прототипа.':`LLM: ${run.usage.model}. Решения требуют подтверждения. Число обращений к API в исходном запуске: ${run.usage.calls}.`;
 const rows=visibleRows();if(!rows.some(r=>r.id===selected))selected=rows[0]?.id||null;$('list').replaceChildren();
 rows.forEach(f=>{const button=node('button',undefined,'record'+(f.id===selected?' active':''));button.setAttribute('role','listitem');button.setAttribute('aria-label',f.text);const top=node('div',undefined,'row-top');top.append(node('span',f.id,'record-id'),node('span',f.human_label?'Проверено человеком':actionNames[f.decision.action],'status '+f.decision.action));button.append(top,node('p',f.text,'record-text'),node('div',labelName(f.label)+(f.decision.action==='change'?' → '+labelName(f.decision.proposed_label):''),'labels'));button.onclick=()=>{selected=f.id;render();};$('list').append(button);});
 if(!rows.length)$('list').append(node('p','В этом разделе нет обращений.','muted'));
 renderDetail(run.findings.find(f=>f.id===selected));
 $('trace').replaceChildren(...run.trace.map(t=>node('div',`${agentNames[t.agent]}${t.record_id?' / '+t.record_id:''}${t.recheck?' / повторная проверка':''}: ${t.result.reason || (t.agent==='instruction'?'Проверена инструкция '+t.result.version:JSON.stringify(t.result).slice(0,300))}`)));
}
function renderDetail(f){
 const d=$('detail');d.replaceChildren();if(!f){d.append(node('p','Очередь завершена. Можно открыть все обращения или загрузить следующий набор.'));return;}
 d.append(node('p','ОБРАЩЕНИЕ / '+f.id,'eyebrow'),node('blockquote',f.text));
 const pair=node('div',undefined,'label-pair');[['Текущая метка',labelName(f.label)],['Рекомендация',f.decision.action==='change'?labelName(f.decision.proposed_label):actionNames[f.decision.action]]].forEach(([key,value])=>{const col=node('div');col.append(node('span',key),node('strong',value));pair.append(col);});d.append(pair);
 d.append(node('h3','Решение арбитра'),node('p',f.decision.reason),node('h3','Основание в тексте'),node('p',f.semantic.evidence,'evidence'),node('p',f.semantic.reason,'muted'));
 d.append(node('h3','Похожие обращения'));
 f.consistency.neighbors.slice(0,2).forEach(n=>{const el=node('div',n.text,'neighbor');el.append(node('small',`${labelName(n.label)} · сходство текстов ${(n.similarity*100).toFixed(0)}% · ${n.id}`));d.append(el);});
 const form=node('div',undefined,'decision-form');const label=node('label','Подтвердите итоговую метку');label.htmlFor='human-label';const select=node('select');select.id='human-label';taxonomy.classes.forEach(c=>{const o=node('option',c.name);o.value=c.id;select.append(o);});select.value=f.decision.proposed_label||f.label;const btn=node('button','Подтвердить','primary');const actions=node('div',undefined,'actions');actions.append(select,btn);form.append(label,actions,node('p','Решение попадёт в журнал. Агент очереди перепроверит связанные обращения.','muted'));if(f.human_label)form.append(node('p','Ранее подтверждено: '+labelName(f.human_label),'saved'));
 btn.onclick=async()=>{btn.disabled=true;$('error').hidden=true;busy(true,'Сохраняем решение и пересматриваем связанную группу…');try{run=await request('/api/feedback',{id:f.id,label:select.value});render();}catch(e){fail(e);btn.disabled=false;}finally{busy(false);}};d.append(form);
}
$('filter').onchange=()=>{selected=null;if(run)render();};
$('demo').onclick=()=>demo().catch(fail);
$('csv-file').onchange=async e=>{const f=e.target.files[0];if(!f)return;if(f.size>2_000_000){fail(Error('Файл больше 2 МБ.'));return;}csv=await f.text();$('file-name').textContent=f.name;$('error').hidden=true;};
$('mode').onchange=()=>{$('disclosure').textContent=$('mode').value==='demo'?'Демо использует эвристики и синтетические обращения. Результаты не измеряют качество LLM.':'При запуске тексты будут отправлены в настроенный LLM API. Адрес и модель задаются через переменные окружения сервера.';};
$('run').onclick=async()=>{$('error').hidden=true;busy(true,$('mode').value==='llm'?'Модель проверяет обращения. Это может занять несколько минут…':'Пять агентов проверяют обращения…');try{run=await request('/api/audit',{csv,mode:$('mode').value,budget:Number($('budget').value)});selected=null;render();}catch(e){fail(e);}finally{busy(false);}};
(async()=>{taxonomy=(await request('/api/info')).taxonomy;await demo();const previous=await request('/api/state');if(previous.findings){run=previous;render();}})().catch(fail);

