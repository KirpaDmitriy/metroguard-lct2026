const form = document.querySelector("#upload-form");
const input = document.querySelector("#bag");
const message = document.querySelector("#message");
const progressCard = document.querySelector("#progress-card");
const progress = document.querySelector("#progress");
const progressLabel = document.querySelector("#progress-label");
const statusLabel = document.querySelector("#status");
const algorithm = document.querySelector("#algorithm");
const results = document.querySelector("#results");
const showCloud = document.querySelector("#show-cloud");
const showDetections = document.querySelector("#show-detections");

const stateLabels = {
  CLEAR: "ПУТЬ СВОБОДЕН",
  UNKNOWN: "НЕДОСТАТОЧНО ДАННЫХ",
  OBSTACLE: "ПРЕПЯТСТВИЕ",
};

const algorithmLabels = {
  geometry: "Только геометрия",
  linear_hybrid: "Геометрия и линейная модель",
  tree_hybrid: "Геометрия и ансамбль деревьев",
};

const reasonLabels = {
  geometry_plus_portable_tree_ranker:
    "Геометрический кандидат подтверждён ансамблем деревьев",
  geometric_components_rejected_by_portable_tree_ranker:
    "Геометрия нашла компоненты, но ансамбль не подтвердил препятствие",
  geometry_plus_risk_model:
    "Геометрический кандидат подтверждён линейной моделью",
  geometric_component_rejected_by_risk_model:
    "Геометрия нашла компонент, но линейная модель его отклонила",
  supported_component_inside_core_clearance:
    "Компонент находится внутри габарита и поддержан точками",
  observable_clearance_without_supported_components:
    "Габарит наблюдается, подтверждённых компонентов нет",
  insufficient_track_observability: "Путь виден недостаточно надёжно",
  rail_pair_not_observed: "Не удалось надёжно найти пару рельсов",
};

const jobLabels = {
  queued: "В очереди",
  running: "Обработка",
  complete: "Готово",
  failed: "Ошибка",
};

let activePreset = null;
let currentItem = null;

input.addEventListener("change", () => {
  document.querySelector("#filename").textContent =
    input.files[0]?.name || "Файл не выбран";
});

algorithm.addEventListener("change", () => {
  if (activePreset) loadPreset(activePreset);
});

showCloud.addEventListener("change", redrawCurrentFrame);
showDetections.addEventListener("change", redrawCurrentFrame);
form.addEventListener("submit", uploadBag);

function redrawCurrentFrame() {
  if (currentItem) drawScene(currentItem);
}

async function readJson(response, fallback) {
  const contentType = response.headers.get("content-type") || "";
  if (!contentType.includes("application/json")) {
    throw new Error(`${fallback}: сервер вернул HTTP ${response.status}`);
  }
  const payload = await response.json();
  if (!response.ok) {
    throw new Error(payload.detail || `${fallback}: HTTP ${response.status}`);
  }
  return payload;
}

async function uploadBag(event) {
  event.preventDefault();
  activePreset = null;
  message.textContent = "";
  results.classList.add("hidden");
  const button = form.querySelector("button");
  button.disabled = true;
  try {
    const body = new FormData();
    body.append("bag", input.files[0]);
    body.append("algorithm", algorithm.value);
    const response = await fetch("/api/jobs", { method: "POST", body });
    const payload = await readJson(response, "Ошибка загрузки");
    progressCard.classList.remove("hidden");
    await watch(payload.job_id);
  } catch (error) {
    message.textContent = error.message;
    button.disabled = false;
  }
}

async function loadPresets() {
  const container = document.querySelector("#presets");
  try {
    const response = await fetch("/api/presets");
    const items = await readJson(response, "Готовые сценарии недоступны");
    container.replaceChildren(...items.map(presetButton));
  } catch (error) {
    container.textContent = error.message;
  }
}

function presetButton(item) {
  const button = document.createElement("button");
  const title = document.createElement("strong");
  const description = document.createElement("span");
  button.type = "button";
  button.className = "preset";
  title.textContent = item.title;
  description.textContent = item.description;
  button.append(title, description);
  button.addEventListener("click", () => loadPreset(item.id));
  return button;
}

async function loadPreset(id) {
  activePreset = id;
  message.textContent = "";
  progressCard.classList.add("hidden");
  results.classList.add("hidden");
  try {
    const [resultResponse, cloudResponse] = await Promise.all([
      fetch(`/api/presets/${id}?algorithm=${algorithm.value}`),
      fetch(`/api/presets/${id}/visualization`),
    ]);
    const payload = await readJson(resultResponse, "Пресет недоступен");
    if (cloudResponse.ok) {
      const cloud = await readJson(cloudResponse, "Визуализация недоступна");
      attachCloud(payload.result.timeline, cloud.frames);
    }
    render(payload.result, payload.preset);
  } catch (error) {
    message.textContent = error.message;
  }
}

function attachCloud(timeline, frames) {
  const byFrame = new Map(frames.map((item) => [item.frame, item]));
  for (const item of timeline) item.visualization = byFrame.get(item.frame);
}

async function watch(jobId) {
  try {
    const response = await fetch(`/api/jobs/${jobId}`);
    const job = await readJson(response, "Статус задания недоступен");
    updateProgress(job);
    if (job.status === "failed") {
      message.textContent = job.error || "Анализ завершился с ошибкой";
      form.querySelector("button").disabled = false;
      return;
    }
    if (job.status === "complete") {
      const resultResponse = await fetch(job.result_url);
      const result = await readJson(resultResponse, "Результат недоступен");
      render(result, null);
      form.querySelector("button").disabled = false;
      return;
    }
    setTimeout(() => watch(jobId), 1000);
  } catch (error) {
    message.textContent = error.message;
    form.querySelector("button").disabled = false;
  }
}

function updateProgress(job) {
  const value = Math.round((job.progress || 0) * 100);
  progress.style.width = `${value}%`;
  progressLabel.textContent = `${value}%`;
  statusLabel.textContent = jobLabels[job.status] || job.status;
}

function render(result, preset) {
  progress.style.width = "100%";
  progressLabel.textContent = "100%";
  statusLabel.textContent = "Готово";
  renderSummary(result);
  renderContext(preset);
  renderVideo(preset);
  renderTimeline(result.timeline);
  results.classList.remove("hidden");
  show(result.timeline.find((item) => item.state === "OBSTACLE") || result.timeline[0]);
}

function renderSummary(result) {
  const summary = result.summary;
  const metrics = [
    ["Алгоритм", algorithmLabels[result.algorithm] || result.algorithm],
    ["Кадров", summary.frames],
    ["Тревог", summary.obstacle_frames],
    [
      "Ближайшая",
      summary.nearest_obstacle_m == null
        ? "—"
        : `${summary.nearest_obstacle_m.toFixed(1)} м`,
    ],
    ["Скорость", `${summary.throughput_fps.toFixed(1)} кадр/с`],
    [
      "Задержка, 95%",
      summary.latency_p95_ms == null
        ? "—"
        : `${summary.latency_p95_ms.toFixed(1)} мс`,
    ],
  ];
  const cards = metrics.map(([name, value]) => {
    const card = document.createElement("div");
    const label = document.createElement("span");
    const number = document.createElement("b");
    card.className = "metric";
    label.textContent = name;
    number.textContent = value;
    card.append(label, number);
    return card;
  });
  document.querySelector("#summary").replaceChildren(...cards);
}

function renderContext(preset) {
  document.querySelector("#result-context").textContent = preset
    ? `${preset.title}. ${preset.source}. Показаны заранее рассчитанные результаты выбранного алгоритма.`
    : "Результат загруженного набора.";
}

function renderVideo(preset) {
  const media = document.querySelector("#media-card");
  const video = document.querySelector("#preset-video");
  if (preset?.video_url) {
    video.src = preset.video_url;
    media.classList.remove("hidden");
    return;
  }
  video.removeAttribute("src");
  media.classList.add("hidden");
}

function renderTimeline(items) {
  const timeline = document.querySelector("#timeline");
  const ticks = items.map((item) => {
    const tick = document.createElement("button");
    tick.className = `tick ${item.state}`;
    tick.title = `Кадр ${item.frame}: ${stateLabels[item.state]}`;
    tick.addEventListener("click", () => show(item));
    return tick;
  });
  timeline.replaceChildren(...ticks);
}

function show(item) {
  if (!item) return;
  currentItem = item;
  const state = document.querySelector("#frame-state");
  const reason =
    reasonLabels[item.reason] ||
    "Решение сформировано по геометрии пути и признакам компонента";
  const distance =
    item.distance_m == null ? "не определено" : `${item.distance_m.toFixed(1)} м`;
  state.className = `state-pill ${item.state}`;
  state.textContent = stateLabels[item.state];
  document.querySelector("#details").textContent = [
    `Кадр ${item.frame}`,
    `Решение: ${stateLabels[item.state]}`,
    `Расстояние: ${distance}`,
    `Уверенность: ${Math.round(item.confidence * 100)}%`,
    `Наблюдаемость: ${Math.round(item.observability * 100)}%`,
    `Причина: ${reason}`,
  ].join("\n");
  drawScene(item);
}

function pointInsideObstacle(point, obstacle) {
  return (
    point[1] >= obstacle.distance_min_m - 0.08 &&
    point[1] <= obstacle.distance_max_m + 0.08 &&
    point[0] >= obstacle.lateral_min_m - 0.08 &&
    point[0] <= obstacle.lateral_max_m + 0.08 &&
    point[2] >= obstacle.height_min_m - 0.08 &&
    point[2] <= obstacle.height_max_m + 0.08
  );
}

function drawScene(item) {
  const canvas = document.querySelector("#scene");
  const context = canvas.getContext("2d");
  const points = item.visualization?.points || [];
  const obstacles = item.obstacles || [];
  const observedMax = Math.max(
    0,
    ...points.map((point) => point[1]),
    ...obstacles.map((obstacle) => obstacle.distance_max_m),
  );
  const range = Math.min(150, Math.max(60, Math.ceil(observedMax / 20) * 20));
  const top = { x: 24, y: 44, width: 450, height: 300 };
  const side = { x: 520, y: 44, width: 450, height: 300 };
  const topX = (lateral) => top.x + ((lateral + 2.4) / 4.8) * top.width;
  const topY = (distance) => top.y + top.height - (distance / range) * top.height;
  const sideX = (distance) => side.x + (distance / range) * side.width;
  const sideY = (height) =>
    side.y + side.height - ((height + 0.25) / 3.5) * side.height;

  context.clearRect(0, 0, canvas.width, canvas.height);
  context.fillStyle = "#08111a";
  context.fillRect(0, 0, canvas.width, canvas.height);
  context.font = "16px system-ui";
  context.fillStyle = "#a9b6c7";
  context.fillText("ВИД СВЕРХУ · РЕАЛЬНЫЕ ТОЧКИ", 24, 28);
  context.fillText("ВИД СБОКУ", 520, 28);
  drawPanels(context, top, side, topX, sideY);
  drawRails(context, top, topX);
  if (showCloud.checked) {
    drawPoints(context, points, obstacles, topX, topY, sideX, sideY);
  }
  if (showDetections.checked) {
    drawObstacles(context, obstacles, topX, topY, sideX, sideY);
  }
  if (!points.length) drawEmptyState(context, top, side);
  drawAxes(context, item, points.length, range, top, side);
}

function drawPanels(context, top, side, topX, sideY) {
  context.strokeStyle = "#263446";
  context.strokeRect(top.x, top.y, top.width, top.height);
  context.strokeRect(side.x, side.y, side.width, side.height);
  context.fillStyle = "#62d6aa12";
  context.fillRect(topX(-1.05), top.y, topX(1.05) - topX(-1.05), top.height);
  context.fillRect(side.x, sideY(3), side.width, sideY(0) - sideY(3));
  context.strokeStyle = "#62d6aa";
  context.setLineDash([7, 6]);
  context.strokeRect(topX(-1.05), top.y, topX(1.05) - topX(-1.05), top.height);
  context.strokeRect(side.x, sideY(3), side.width, sideY(0) - sideY(3));
  context.setLineDash([]);
}

function drawRails(context, top, topX) {
  context.strokeStyle = "#65788e";
  context.beginPath();
  context.moveTo(topX(-0.76), top.y);
  context.lineTo(topX(-0.76), top.y + top.height);
  context.moveTo(topX(0.76), top.y);
  context.lineTo(topX(0.76), top.y + top.height);
  context.stroke();
}

function drawPoints(context, points, obstacles, topX, topY, sideX, sideY) {
  for (const point of points) {
    const hit =
      showDetections.checked &&
      obstacles.some((obstacle) => pointInsideObstacle(point, obstacle));
    const radius = hit ? 2.4 : 1.25;
    context.fillStyle = hit ? "#ff6577" : "#72a9d0aa";
    drawPoint(context, topX(point[0]), topY(point[1]), radius);
    drawPoint(context, sideX(point[1]), sideY(point[2]), radius);
  }
}

function drawPoint(context, x, y, radius) {
  context.beginPath();
  context.arc(x, y, radius, 0, Math.PI * 2);
  context.fill();
}

function drawObstacles(context, obstacles, topX, topY, sideX, sideY) {
  context.strokeStyle = "#ff6577";
  context.fillStyle = "#ff657722";
  context.lineWidth = 2.5;
  for (const obstacle of obstacles) {
    drawBox(
      context,
      topX(obstacle.lateral_min_m),
      topY(obstacle.distance_max_m),
      topX(obstacle.lateral_max_m),
      topY(obstacle.distance_min_m),
    );
    drawBox(
      context,
      sideX(obstacle.distance_min_m),
      sideY(obstacle.height_max_m),
      sideX(obstacle.distance_max_m),
      sideY(obstacle.height_min_m),
    );
  }
  context.lineWidth = 1;
}

function drawBox(context, x0, y0, x1, y1) {
  const width = Math.max(4, x1 - x0);
  const height = Math.max(4, y1 - y0);
  context.fillRect(x0, y0, width, height);
  context.strokeRect(x0, y0, width, height);
}

function drawEmptyState(context, top, side) {
  context.fillStyle = "#8999aa";
  context.fillText(
    "Для этого кадра облако не сохранено",
    top.x + 82,
    top.y + 150,
  );
  context.fillText(
    "Доступна геометрия решения",
    side.x + 112,
    side.y + 150,
  );
}

function drawAxes(context, item, pointCount, range, top, side) {
  context.fillStyle = "#718196";
  context.font = "13px system-ui";
  context.fillText("0 м", top.x + 4, top.y + top.height - 6);
  context.fillText(`${range} м`, top.x + 4, top.y + 14);
  context.fillText("0 м", side.x + 4, side.y + side.height - 6);
  context.fillText(
    `${range} м`,
    side.x + side.width - 48,
    side.y + side.height - 6,
  );
  context.fillText("3 м", side.x + 5, side.y + 34);
  context.fillText(
    `Кадр ${item.frame} · ${pointCount} точек для визуализации`,
    24,
    376,
  );
}

loadPresets();
