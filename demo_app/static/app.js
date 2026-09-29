const form=document.querySelector('#upload-form');
const input=document.querySelector('#bag');
const message=document.querySelector('#message');
const progressCard=document.querySelector('#progress-card');
const progress=document.querySelector('#progress');
const progressLabel=document.querySelector('#progress-label');
const statusLabel=document.querySelector('#status');
const algorithm=document.querySelector('#algorithm');
const stateLabels={CLEAR:'ПУТЬ СВОБОДЕН',UNKNOWN:'НЕДОСТАТОЧНО ДАННЫХ',OBSTACLE:'ПРЕПЯТСТВИЕ'};
const algorithmLabels={geometry:'Только геометрия',linear_hybrid:'Геометрия и линейная модель',tree_hybrid:'Геометрия и ансамбль деревьев'};
const reasonLabels={geometry_plus_portable_tree_ranker:'Геометрический кандидат подтверждён ансамблем деревьев',geometric_components_rejected_by_portable_tree_ranker:'Геометрия нашла компоненты, но ансамбль не подтвердил препятствие',geometry_plus_risk_model:'Геометрический кандидат подтверждён линейной моделью',geometric_component_rejected_by_risk_model:'Геометрия нашла компонент, но линейная модель его отклонила',supported_component_inside_core_clearance:'Компонент находится внутри габарита и поддержан точками',observable_clearance_without_supported_components:'Габарит наблюдается, подтверждённых компонентов нет',insufficient_track_observability:'Путь виден недостаточно надёжно',rail_pair_not_observed:'Не удалось надёжно найти пару рельсов'};
let activePreset=null;

input.addEventListener('change',()=>document.querySelector('#filename').textContent=input.files[0]?.name||'Файл не выбран');
algorithm.addEventListener('change',()=>{if(activePreset)loadPreset(activePreset)});

form.addEventListener('submit',async event=>{
  event.preventDefault();activePreset=null;message.textContent='';document.querySelector('#results').classList.add('hidden');
  const button=form.querySelector('button');button.disabled=true;
  try{
    const body=new FormData();body.append('bag',input.files[0]);body.append('algorithm',algorithm.value);
    const response=await fetch('/api/jobs',{method:'POST',body});const payload=await response.json();
    if(!response.ok)throw new Error(payload.detail||'Ошибка загрузки');
    progressCard.classList.remove('hidden');await watch(payload.job_id);
  }catch(error){message.textContent=error.message;button.disabled=false}
});

async function loadPresets(){
  const container=document.querySelector('#presets');
  try{
    const items=await fetch('/api/presets').then(response=>response.json());container.innerHTML='';
    for(const item of items){const button=document.createElement('button');button.type='button';button.className='preset';button.innerHTML=`<strong>${item.title}</strong><span>${item.description}</span>`;button.addEventListener('click',()=>loadPreset(item.id));container.appendChild(button)}
  }catch(_error){container.textContent='Не удалось загрузить готовые сценарии.'}
}

async function loadPreset(id){
  activePreset=id;message.textContent='';progressCard.classList.add('hidden');document.querySelector('#results').classList.add('hidden');
  try{const response=await fetch(`/api/presets/${id}?algorithm=${algorithm.value}`);const payload=await response.json();if(!response.ok)throw new Error(payload.detail||'Пресет недоступен');render(payload.result,payload.preset)}catch(error){message.textContent=error.message}
}

async function watch(jobId){
  const response=await fetch(`/api/jobs/${jobId}`);const job=await response.json();
  const value=Math.round((job.progress||0)*100);progress.style.width=`${value}%`;progressLabel.textContent=`${value}%`;statusLabel.textContent=labels[job.status]||job.status;
  if(job.status==='failed'){message.textContent=job.error||'Анализ завершился с ошибкой';form.querySelector('button').disabled=false;return}
  if(job.status==='complete'){const result=await fetch(job.result_url).then(item=>item.json());render(result,null);form.querySelector('button').disabled=false;return}
  setTimeout(()=>watch(jobId),1000);
}

const labels={queued:'В очереди',running:'Обработка',complete:'Готово',failed:'Ошибка'};

function render(result,preset){
  progress.style.width='100%';progressLabel.textContent='100%';statusLabel.textContent='Готово';
  const s=result.summary;const metrics=[['Алгоритм',algorithmLabels[result.algorithm]||result.algorithm],['Кадров',s.frames],['Тревог',s.obstacle_frames],['Ближайшая',s.nearest_obstacle_m==null?'—':`${s.nearest_obstacle_m.toFixed(1)} м`],['Скорость',`${s.throughput_fps.toFixed(1)} кадр/с`],['Задержка, 95%',s.latency_p95_ms==null?'—':`${s.latency_p95_ms.toFixed(1)} мс`]];
  document.querySelector('#summary').innerHTML=metrics.map(([name,value])=>`<div class="metric"><span>${name}</span><b>${value}</b></div>`).join('');
  document.querySelector('#result-context').textContent=preset?`${preset.title}. ${preset.source}. Показаны заранее рассчитанные результаты выбранного алгоритма.`:'Результат загруженного набора.';
  const media=document.querySelector('#media-card');const video=document.querySelector('#preset-video');
  if(preset?.video_url){video.src=preset.video_url;media.classList.remove('hidden')}else{video.removeAttribute('src');media.classList.add('hidden')}
  const timeline=document.querySelector('#timeline');timeline.innerHTML='';
  for(const item of result.timeline){const tick=document.createElement('button');tick.className=`tick ${item.state}`;tick.title=`Кадр ${item.frame}: ${stateLabels[item.state]}`;tick.addEventListener('click',()=>show(item));timeline.appendChild(tick)}
  document.querySelector('#results').classList.remove('hidden');show(result.timeline.find(item=>item.state==='OBSTACLE')||result.timeline[0]);
}

function show(item){
  if(!item)return;
  const state=document.querySelector('#frame-state');state.className=`state-pill ${item.state}`;state.textContent=stateLabels[item.state];
  const reason=reasonLabels[item.reason]||'Решение сформировано по геометрии пути и признакам компонента';const distance=item.distance_m==null?'не определено':`${item.distance_m.toFixed(1)} м`;
  document.querySelector('#details').textContent=`Кадр ${item.frame}\nРешение: ${stateLabels[item.state]}\nРасстояние: ${distance}\nУверенность: ${Math.round(item.confidence*100)}%\nНаблюдаемость: ${Math.round(item.observability*100)}%\nПричина: ${reason}`;drawScene(item);
}

function drawScene(item){
  const canvas=document.querySelector('#scene');const ctx=canvas.getContext('2d');const h=canvas.height;ctx.clearRect(0,0,canvas.width,h);ctx.fillStyle='#08111a';ctx.fillRect(0,0,canvas.width,h);ctx.font='16px system-ui';ctx.fillStyle='#a9b6c7';ctx.fillText('ВИД СВЕРХУ',24,28);ctx.fillText('ВИД СБОКУ',520,28);
  const top={x:24,y:44,w:450,h:260};const side={x:520,y:44,w:450,h:260};ctx.strokeStyle='#263446';ctx.strokeRect(top.x,top.y,top.w,top.h);ctx.strokeRect(side.x,side.y,side.w,side.h);
  const center=top.x+top.w/2;const corridorHalf=55;ctx.fillStyle='#62d6aa18';ctx.fillRect(center-corridorHalf,top.y,corridorHalf*2,top.h);ctx.strokeStyle='#62d6aa';ctx.setLineDash([7,6]);ctx.strokeRect(center-corridorHalf,top.y,corridorHalf*2,top.h);ctx.fillStyle='#62d6aa18';ctx.fillRect(side.x,side.y+20,side.w,side.h-20);ctx.strokeRect(side.x,side.y+20,side.w,side.h-20);ctx.setLineDash([]);
  ctx.strokeStyle='#65788e';ctx.beginPath();ctx.moveTo(center-20,top.y);ctx.lineTo(center-20,top.y+top.h);ctx.moveTo(center+20,top.y);ctx.lineTo(center+20,top.y+top.h);ctx.stroke();
  const obstacles=item.obstacles||[];
  for(const obstacle of obstacles){const d0=Math.min(100,Math.max(0,obstacle.distance_min_m));const d1=Math.min(100,Math.max(0,obstacle.distance_max_m));const x0=center+obstacle.lateral_min_m*52;const x1=center+obstacle.lateral_max_m*52;const y0=top.y+top.h-d1/100*top.h;const y1=top.y+top.h-d0/100*top.h;ctx.fillStyle='#ff657799';ctx.fillRect(x0,y0,Math.max(4,x1-x0),Math.max(4,y1-y0));const sx0=side.x+d0/100*side.w;const sx1=side.x+d1/100*side.w;const sy0=side.y+side.h-Math.min(3.2,obstacle.height_max_m)/3.2*side.h;const sy1=side.y+side.h-Math.max(0,obstacle.height_min_m)/3.2*side.h;ctx.fillRect(sx0,sy0,Math.max(4,sx1-sx0),Math.max(4,sy1-sy0))}
  if(!obstacles.length){ctx.fillStyle='#8999aa';ctx.fillText('Подтверждённых компонентов нет',top.x+85,top.y+135);ctx.fillText('Алгоритм не подтверждает препятствие',side.x+92,side.y+135)}
  ctx.fillStyle='#718196';ctx.font='13px system-ui';ctx.fillText('0 м',top.x+4,top.y+top.h-6);ctx.fillText('100 м',top.x+4,top.y+14);ctx.fillText('0 м',side.x+4,side.y+top.h-6);ctx.fillText('100 м',side.x+side.w-42,side.y+top.h-6);ctx.fillText('3 м',side.x+5,side.y+34);
}

loadPresets();
