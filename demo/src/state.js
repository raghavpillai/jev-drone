import { HouseView } from './scene.js';

const $ = id => document.getElementById(id);
const title = text => String(text || '—').replaceAll('_', ' ').replace(/\b\w/g, c => c.toUpperCase());
const clock = seconds => `${Math.floor(seconds / 60).toString().padStart(2, '0')}:${Math.floor(seconds % 60).toString().padStart(2, '0')}`;
const signed = value => Number.isFinite(value) ? `${value > 0 ? '+' : ''}${value.toFixed(2)}` : '—';
const coordinates = point => point ? point.map(v => v.toFixed(2)).join(' / ') : '—';
const roomAt = point => point && Object.entries(episode.world.rooms).find(([, [lo, hi]]) => point[0] >= lo[0] && point[0] <= hi[0] && point[1] >= lo[1] && point[1] <= hi[1])?.[0];
const view = new HouseView($('stage'));
function resize() {
  const width = $('stage').clientWidth;
  view.renderer.setSize(width, window.innerHeight);
  view.camera.aspect = width / window.innerHeight;
  view.updateLens();
}
window.addEventListener('resize', resize); resize();

const channels = [
  ['forward_mps', 'Forward / back', 'BACK', 'FORWARD', .5, 'm/s'],
  ['right_mps', 'Strafe', 'LEFT', 'RIGHT', .5, 'm/s'],
  ['up_mps', 'Vertical', 'DOWN', 'UP', .25, 'm/s'],
  ['yaw_rate_rps', 'Pan', 'RIGHT', 'LEFT', .6, 'rad/s'],
];
for (const [key, label, negative, positive] of channels) {
  const row = document.createElement('div'); row.className = 'axis'; row.dataset.axis = key;
  row.innerHTML = `<span class="axis-label">${label}</span><div><div class="axis-track"><i></i></div><div class="axis-direction"><span>${negative}</span><span>${positive}</span></div></div><b class="axis-value">0.00</b>`;
  $('axes').append(row);
}
const directions = [['forward', 'Front'], ['backward', 'Rear'], ['left', 'Left'], ['right', 'Right'], ['up', 'Up'], ['down', 'Down']];
for (const [key, label] of directions) {
  const cell = document.createElement('div'); cell.dataset.direction = key;
  cell.innerHTML = `<span>${label}</span><b>—</b>`; $('clearance').append(cell);
}

// All streams use mission-relative simulator time. A selection is visible only
// after its recorded response; applied controls also include lease expirations.
function latest(stream, time) {
  let lo = 0, hi = stream.length;
  while (lo < hi) { const mid = (lo + hi) >>> 1; if (stream[mid].time <= time) lo = mid + 1; else hi = mid; }
  return lo ? stream[lo - 1] : null;
}
const manualRendering = new URLSearchParams(location.search).has('render');
let data, episode, planner = [], controller = [], playing = false, rate = 12, time = 0, previous = performance.now(), ready = false;
let missionEnd = 0, missionObjective, foundReport;
let lastPlan, lastLocal, lastControl;

function renderPlanner(plan) {
  $('plan-choice').textContent = title(plan?.choice || 'Awaiting Jev');
  $('plan-room').textContent = title(plan?.room);
  const objective = plan?.objective;
  $('plan-destination').textContent = objective?.dock ? `Launch / ${roomAt(objective.dock) || 'doorway'}` : title(objective?.room);
  const score = plan?.probabilities[plan.choice];
  $('plan-mode').textContent = title(plan?.task_kind) + (score != null ? ` · ${Math.round(score * 100)}%` : '');
  $('plan-waypoint').textContent = plan?.option.position ? `Waypoint  ${coordinates(plan.option.position)} m` : plan?.option.look ? `Look ${plan.option.look}` : 'No position requested';
  $('views').textContent = plan?.viewpoints_total ? `${plan.viewpoints_seen} / ${plan.viewpoints_total}` : 'No survey yet';
  $('headings').textContent = plan ? `${plan.heading_sectors.length} / 8 sectors` : '—';
  $('evidence').textContent = !plan ? '—' : objective?.dock ? 'Returning to launch' : plan.ready ? 'Ready to inspect' : plan.visible ? 'Visible in front camera' : 'Not currently visible';
  $('history').replaceChildren();
  for (const task of plan?.recent_tasks || []) {
    const row = document.createElement('div'), name = document.createElement('span'), outcome = document.createElement('b');
    name.textContent = title(task.name); outcome.textContent = title(task.outcome);
    outcome.classList.toggle('failed', task.outcome !== 'arrived'); row.append(name, outcome); $('history').append(row);
  }
}
function renderLocal(local) {
  $('local-choice').textContent = title(local?.choice || 'Awaiting Jev');
  $('accepted').textContent = !local ? '—' : local.accepted ? 'APPLIED' : 'NOT APPLIED';
  $('accepted').classList.toggle('rejected', Boolean(local && !local.accepted));
  $('local-task').textContent = local?.task?.name ? `Task: ${local.task.name}` : 'No control decision yet';
  $('offset').textContent = local?.offset ? ['forward', 'right', 'up'].map(k => signed(local.offset[k])).join(' / ') + ' m' : '—';
  $('heading').textContent = local?.heading_error != null ? `${signed(local.heading_error)}°` : '—';
  $('latency').textContent = local ? `${Math.round(local.latency_ms)} ms` : '—';
  for (const [key] of directions) {
    const cell = document.querySelector(`[data-direction="${key}"]`), value = local?.clearance[key];
    cell.querySelector('b').textContent = value == null ? '?' : value.toFixed(1);
    cell.classList.toggle('unknown', value == null);
    cell.classList.toggle('limited', Boolean(local?.depth_status[key] && local.depth_status[key] !== 'OPEN'));
  }
  $('scores').replaceChildren();
  const probabilities = Object.entries(local?.probabilities || {}).sort((a, b) => b[1] - a[1]).slice(0, 3);
  for (const [name, probability] of probabilities) {
    const row = document.createElement('div'); row.className = 'score'; row.classList.toggle('selected', name === local.choice);
    const label = document.createElement('span'), track = document.createElement('div'), fill = document.createElement('i'), value = document.createElement('b');
    label.textContent = title(name); track.className = 'score-track'; fill.style.width = `${probability * 100}%`;
    value.textContent = `${Math.round(probability * 100)}%`; track.append(fill); row.append(label, track, value); $('scores').append(row);
  }
}
function renderControls(control) {
  $('control-source').textContent = control?.source || 'Stopped';
  for (const [key, , , , maximum, unit] of channels) {
    const value = control?.values[key] || 0, row = document.querySelector(`[data-axis="${key}"]`), fill = row.querySelector('i');
    row.querySelector('.axis-value').textContent = `${signed(value)} ${unit}`;
    const width = Math.min(1, Math.abs(value) / maximum) * 50;
    fill.style.left = `${value < 0 ? 50 - width : 50}%`; fill.style.width = `${width}%`;
    row.dataset.value = value;
  }
  const short = { forward_mps: 'forward', right_mps: 'strafe', up_mps: 'vertical', yaw_rate_rps: 'pan' };
  $('patch').textContent = !control ? 'Waiting for the first update' : control.source !== 'Jev update' ? `${control.source} · all axes zero`
    : Object.keys(control.patch || {}).length ? `Changed: ${Object.entries(control.patch).map(([key, value]) => `${short[key]} ${signed(value)}`).join(' · ')}` : 'No axis changes · lease renewed';
}
function draw(dt, presentationTime, animate = false) {
  const point = latest(episode.trace, time) || episode.trace[0];
  const index = episode.trace.indexOf(point), next = episode.trace[Math.min(index + 1, episode.trace.length - 1)];
  const fraction = next.time > point.time ? Math.max(0, Math.min(1, (time - point.time) / (next.time - point.time))) : 0;
  view.draw(point, next, fraction, index, dt, playing || animate, presentationTime);
  const plan = latest(planner, time), local = latest(controller, time), control = latest(data.controls, time);
  if (plan !== lastPlan) { renderPlanner(plan); lastPlan = plan; }
  if (local !== lastLocal) { renderLocal(local); lastLocal = local; }
  if (control !== lastControl) { renderControls(control); lastControl = control; }
  $('plan-age').textContent = plan ? `Snapshot · ${(time - plan.request_time).toFixed(1)}s ago` : 'Awaiting decision';
  $('local-age').textContent = local ? `Snapshot · ${(time - local.request_time).toFixed(1)}s ago` : 'Awaiting decision';
  const completed = Boolean(foundReport && foundReport.time <= time);
  const objective = missionObjective;
  const currentRoom = roomAt(point.position);
  const phase = completed ? 'Item found' : time >= missionEnd ? 'Search unfinished' : currentRoom !== objective?.room ? `Go to ${objective?.room}` : local?.visible ? 'Item spotted · verifying' : `Searching ${objective.room}`;
  $('mission-phase').textContent = phase;
  $('mission-detail').textContent = `${title(currentRoom || 'Doorway')} · ${completed ? 'Search complete' : 'Search in progress'}`;
  $('time').textContent = clock(time);
  $('playback-note').textContent = `Recorded simulation · ${playing ? `${rate}× replay` : 'paused'}`;
}
function frame(now) {
  const dt = Math.min((now - previous) / 1000, .25); previous = now;
  if (ready) {
    if (playing) { time = Math.min(missionEnd, time + dt * rate); if (time === missionEnd) playing = false; }
    draw(dt);
  }
  requestAnimationFrame(frame);
}
window.stateReplay = {
  get ready() { return ready; },
  get status() { return { time, duration: missionEnd, playing, controls: latest(data?.controls || [], time), planner: lastPlan, local: lastLocal, wallHeight: view.cutawayHeight, targetHighlighted: Boolean(view.targetEffect) }; },
  play(speed = 12) { rate = speed; playing = true; previous = performance.now(); },
  pause() { playing = false; },
  seek(value) { playing = false; time = Math.max(0, Math.min(missionEnd, value)); view.snap = true; draw(1 / 30, time / rate); },
  renderFrame(index, fps = 30, speed = 12) {
    if (!manualRendering) throw new Error('Frame export requires ?render');
    rate = speed; time = Math.min(missionEnd, index * speed / fps); playing = time < missionEnd;
    if (!index) view.snap = true;
    draw(1 / fps, index / fps, true);
    $('playback-note').textContent = `Recorded search · ${speed}× · ${fps} fps`;
    view.renderer.getContext().finish();
    return { frame: index, time, camera: view.camera.position.toArray(), drone: view.drone.position.toArray() };
  },
};
try {
  const response = await fetch('data/state-replay.json'); if (!response.ok) throw new Error('State replay data unavailable'); data = await response.json();
  const episodeResponse = await fetch(data.episode); if (!episodeResponse.ok) throw new Error('Flight recording unavailable'); episode = await episodeResponse.json();
  const stage = episode.world.mission.findIndex(objective => objective.marker);
  if (stage < 0) throw new Error('This view requires a recorded item-search objective');
  missionObjective = episode.world.mission[stage];
  foundReport = episode.reports.find(report => report.valid && report.stage === stage);
  missionEnd = foundReport?.time ?? episode.duration;
  $('mission-objective').textContent = `Find the ${missionObjective.marker} item`;
  $('mission-instruction').textContent = `Search the ${missionObjective.room}. Finish when found.`;
  planner = data.decisions.filter(d => d.role === 'planner'); controller = data.decisions.filter(d => d.role === 'control');
  view.load(episode); resize(); view.setMode('chase', true);
  renderPlanner(null); renderLocal(null); renderControls(null); ready = true; draw(1 / 30);
} catch (error) { $('mission-phase').textContent = error.message; throw error; }
if (!manualRendering) {
  window.stateReplay.play();
  window.addEventListener('keydown', event => {
    if (event.code === 'Space') {
      event.preventDefault();
      playing ? window.stateReplay.pause() : window.stateReplay.play();
    } else if (event.code === 'KeyR') {
      window.stateReplay.seek(0);
      window.stateReplay.play();
    }
  });
  requestAnimationFrame(frame);
}
