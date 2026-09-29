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
let currentItem=null;

input.addEventListener('change',()=>document.querySelector('#filename').textContent=input.files[0]?.name||'Файл не выбран');
algorithm.addEventListener('change',()=>{if(activePreset)loadPreset(activePreset)});
document.querySelector('#show-cloud').addEventListener('change',()=>{if(currentItem)drawScene(currentItem)});
document.querySelector('#show-detections').addEventListener('change',()=>{if(currentItem)drawScene(currentItem)});

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
  try{
    const [response,visualResponse]=await Promise.all([fetch(`/api/presets/${id}?algorithm=${algorithm.value}`),fetch(`/api/presets/${id}/visualization`)]);
    const payload=await response.json();if(!response.ok)throw new Error(payload.detail||'Пресет недоступен');
    if(visualResponse.ok){const visual=await visualResponse.json();const byFrame=new Map(visual.frames.map(item=>[item.frame,item]));for(const item of payload.result.timeline){item.visualization=byFrame.get(item.frame)}}
    render(payload.result,payload.preset)
  }catch(error){message.textContent=error.message}
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
  currentItem=item;
  const state=document.querySelector('#frame-state');state.className=`state-pill ${item.state}`;state.textContent=stateLabels[item.state];
  const reason=reasonLabels[item.reason]||'Решение сформировано по геометрии пути и признакам компонента';const distance=item.distance_m==null?'не определено':`${item.distance_m.toFixed(1)} м`;
  document.querySelector('#details').textContent=`Кадр ${item.frame}\nРешение: ${stateLabels[item.state]}\nРасстояние: ${distance}\nУверенность: ${Math.round(item.confidence*100)}%\nНаблюдаемость: ${Math.round(item.observability*100)}%\nПричина: ${reason}`;drawScene(item);
}

function drawScene(item){
  const canvas=document.querySelector('#scene');const ctx=canvas.getContext('2d');const points=item.visualization?.points||[];const obstacles=item.obstacles||[];const showCloud=document.querySelector('#show-cloud').checked;const showDetections=document.querySelector('#show-detections').checked;
  const observedMax=Math.max(0,...points.map(point=>point[1]),...obstacles.map(value=>value.distance_max_m));const range=Math.min(150,Math.max(60,Math.ceil(observedMax/20)*20));
  ctx.clearRect(0,0,canvas.width,canvas.height);ctx.fillStyle='#08111a';ctx.fillRect(0,0,canvas.width,canvas.height);ctx.font='16px system-ui';ctx.fillStyle='#a9b6c7';ctx.fillText('ВИД СВЕРХУ · РЕАЛЬНЫЕ ТОЧКИ',24,28);ctx.fillText('ВИД СБОКУ',520,28);
  const top={x:24,y:44,w:450,h:300};const side={x:520,y:44,w:450,h:300};ctx.strokeStyle='#263446';ctx.strokeRect(top.x,top.y,top.w,top.h);ctx.strokeRect(side.x,side.y,side.w,side.h);
  const topX=lateral=>top.x+(lateral+2.4)/4.8*top.w;const topY=distance=>top.y+top.h-distance/range*top.h;const sideX=distance=>side.x+distance/range*side.w;const sideY=height=>side.y+side.h-(height+.25)/3.5*side.h;
  ctx.fillStyle='#62d6aa12';ctx.fillRect(topX(-1.05),top.y,topX(1.05)-topX(-1.05),top.h);ctx.fillRect(side.x,sideY(3),side.w,sideY(0)-sideY(3));ctx.strokeStyle='#62d6aa';ctx.setLineDash([7,6]);ctx.strokeRect(topX(-1.05),top.y,topX(1.05)-topX(-1.05),top.h);ctx.strokeRect(side.x,sideY(3),side.w,sideY(0)-sideY(3));ctx.setLineDash([]);
  ctx.strokeStyle='#65788e';ctx.beginPath();ctx.moveTo(topX(-.76),top.y);ctx.lineTo(topX(-.76),top.y+top.h);ctx.moveTo(topX(.76),top.y);ctx.lineTo(topX(.76),top.y+top.h);ctx.stroke();
  const inside=(point,obstacle)=>point[1]>=obstacle.distance_min_m-.08&&point[1]<=obstacle.distance_max_m+.08&&point[0]>=obstacle.lateral_min_m-.08&&point[0]<=obstacle.lateral_max_m+.08&&point[2]>=obstacle.height_min_m-.08&&point[2]<=obstacle.height_max_m+.08;
  if(showCloud){for(const point of points){const hit=showDetections&&obstacles.some(obstacle=>inside(point,obstacle));ctx.fillStyle=hit?'#ff6577':'#72a9d0aa';const radius=hit?2.4:1.25;ctx.beginPath();ctx.arc(topX(point[0]),topY(point[1]),radius,0,Math.PI*2);ctx.fill();ctx.beginPath();ctx.arc(sideX(point[1]),sideY(point[2]),radius,0,Math.PI*2);ctx.fill()}}
  if(showDetections){for(const obstacle of obstacles){ctx.strokeStyle='#ff6577';ctx.lineWidth=2.5;ctx.fillStyle='#ff657722';const tx0=topX(obstacle.lateral_min_m);const tx1=topX(obstacle.lateral_max_m);const ty0=topY(obstacle.distance_max_m);const ty1=topY(obstacle.distance_min_m);ctx.fillRect(tx0,ty0,Math.max(4,tx1-tx0),Math.max(4,ty1-ty0));ctx.strokeRect(tx0,ty0,Math.max(4,tx1-tx0),Math.max(4,ty1-ty0));const sx0=sideX(obstacle.distance_min_m);const sx1=sideX(obstacle.distance_max_m);const sy0=sideY(obstacle.height_max_m);const sy1=sideY(obstacle.height_min_m);ctx.fillRect(sx0,sy0,Math.max(4,sx1-sx0),Math.max(4,sy1-sy0));ctx.strokeRect(sx0,sy0,Math.max(4,sx1-sx0),Math.max(4,sy1-sy0))}ctx.lineWidth=1}
  if(!points.length){ctx.fillStyle='#8999aa';ctx.fillText('Для этого кадра облако не сохранено',top.x+82,top.y+150);ctx.fillText('Доступна геометрия решения',side.x+112,side.y+150)}
  ctx.fillStyle='#718196';ctx.font='13px system-ui';ctx.fillText('0 м',top.x+4,top.y+top.h-6);ctx.fillText(`${range} м`,top.x+4,top.y+14);ctx.fillText('0 м',side.x+4,side.y+top.h-6);ctx.fillText(`${range} м`,side.x+side.w-48,side.y+top.h-6);ctx.fillText('3 м',side.x+5,side.y+34);ctx.fillText(`Кадр ${item.frame} · ${points.length} точек для визуализации`,24,376);
}

loadPresets();
