const form=document.querySelector('#upload-form');
const input=document.querySelector('#bag');
const message=document.querySelector('#message');
const progressCard=document.querySelector('#progress-card');
const progress=document.querySelector('#progress');
const progressLabel=document.querySelector('#progress-label');
const statusLabel=document.querySelector('#status');
input.addEventListener('change',()=>document.querySelector('#filename').textContent=input.files[0]?.name||'Файл не выбран');
form.addEventListener('submit',async event=>{
  event.preventDefault();message.textContent='';document.querySelector('#results').classList.add('hidden');
  const button=form.querySelector('button');button.disabled=true;
  try{
    const body=new FormData();body.append('bag',input.files[0]);body.append('algorithm',document.querySelector('#algorithm').value);
    const response=await fetch('/api/jobs',{method:'POST',body});const payload=await response.json();
    if(!response.ok)throw new Error(payload.detail||'Ошибка загрузки');
    progressCard.classList.remove('hidden');await watch(payload.job_id);
  }catch(error){message.textContent=error.message;button.disabled=false}
});
async function watch(jobId){
  const response=await fetch(`/api/jobs/${jobId}`);const job=await response.json();
  const value=Math.round((job.progress||0)*100);progress.style.width=`${value}%`;progressLabel.textContent=`${value}%`;statusLabel.textContent=labels[job.status]||job.status;
  if(job.status==='failed'){message.textContent=job.error||'Анализ завершился с ошибкой';form.querySelector('button').disabled=false;return}
  if(job.status==='complete'){const result=await fetch(job.result_url).then(item=>item.json());render(result);form.querySelector('button').disabled=false;return}
  setTimeout(()=>watch(jobId),1000);
}
const labels={queued:'В очереди',running:'Обработка',complete:'Готово',failed:'Ошибка'};
function render(result){
  progress.style.width='100%';progressLabel.textContent='100%';statusLabel.textContent='Готово';
  const s=result.summary;const metrics=[['Алгоритм',result.algorithm],['Кадров',s.frames],['Тревог',s.obstacle_frames],['Ближайшая',s.nearest_obstacle_m==null?'—':`${s.nearest_obstacle_m.toFixed(1)} м`],['Скорость',`${s.throughput_fps.toFixed(1)} FPS`],['p95',s.latency_p95_ms==null?'—':`${s.latency_p95_ms.toFixed(1)} мс`]];
  document.querySelector('#summary').innerHTML=metrics.map(([name,value])=>`<div class="metric"><span>${name}</span><b>${value}</b></div>`).join('');
  const timeline=document.querySelector('#timeline');timeline.innerHTML='';
  for(const item of result.timeline){const tick=document.createElement('button');tick.className=`tick ${item.state}`;tick.title=`Кадр ${item.frame}: ${item.state}`;tick.addEventListener('click',()=>show(item));timeline.appendChild(tick)}
  document.querySelector('#results').classList.remove('hidden');
}
function show(item){document.querySelector('#details').textContent=JSON.stringify({frame:item.frame,state:item.state,distance_m:item.distance_m,confidence:Number(item.confidence.toFixed(3)),observability:Number(item.observability.toFixed(3)),latency_ms:Number(item.latency_ms.toFixed(1)),reason:item.reason,obstacles:item.obstacles},null,2)}
