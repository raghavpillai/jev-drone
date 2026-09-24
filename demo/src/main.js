import { HouseView } from './scene.js';

const $ = id => document.getElementById(id);
const view = new HouseView($('viewport'));
let episode, playback = 0, playing = false, previous = performance.now(), active = 0, generation = 0;
const clock = seconds => `${Math.floor(seconds / 60).toString().padStart(2, '0')}:${Math.floor(seconds % 60).toString().padStart(2, '0')}`;
const title = value => value.replaceAll('_', ' ').replace(/\b\w/g, c => c.toUpperCase());
function setPlaying(value) {
  playing = value && episode?.trace.length > 1;
  $('play').textContent = playing ? 'Ⅱ' : '▶'; $('play').setAttribute('aria-label', playing ? 'Pause replay' : 'Play replay');
}
function setView(mode) {
  view.setMode(mode);
  document.querySelectorAll('[data-view]').forEach(b => { b.classList.toggle('active', b.dataset.view === mode); b.setAttribute('aria-pressed', String(b.dataset.view === mode)); });
  $('view-note').textContent = { chase: 'Recorded simulation', orbit: 'Drag to orbit · right-drag to pan · scroll to zoom', fpv: 'High-resolution reconstruction · not the recorded sensor feed' }[mode];
}
async function load(path) {
  const request = ++generation; setPlaying(false); $('loading').classList.remove('hidden');
  const response = await fetch(path); if (!response.ok) throw new Error(`Flight unavailable (${response.status})`);
  const data = await response.json(); if (request !== generation) return;
  episode = data; playback = 0; active = 0; view.load(episode);
  $('room-count').textContent = `${Object.keys(episode.world.rooms).length} ROOMS`;
  const area = Object.values(episode.world.rooms).reduce((sum, [lo, hi]) => sum + (hi[0] - lo[0]) * (hi[1] - lo[1]), 0);
  $('area').textContent = `${area} M²`;
  const targets = episode.world.mission.filter(o => o.marker), target = targets[0];
  $('mission').textContent = episode.status === 'scene_preview' ? 'Explore the house' : targets.length > 1 ? `Find ${targets.length} markers` : target ? `Find the ${target.marker} marker` : 'Return to launch';
  $('mission-route').textContent = episode.status === 'scene_preview' ? 'Drag to look around the rooms.' : target ? `${targets.map(o => title(o.room)).join(' → ')}${episode.world.mission.some(o => o.dock) ? ' · then return to launch' : ''}` : 'Fly back to the starting position.';
  document.querySelector('.steps').hidden = episode.status === 'scene_preview';
  const safetyFailure = episode.contacts > 0 || episode.violations > 0;
  $('outcome').textContent = episode.status === 'success'
    ? (safetyFailure ? 'COMPLETE · SAFETY FAIL' : 'FINAL · COMPLETE')
    : title(episode.status).toUpperCase();
  $('outcome').classList.toggle('failed', safetyFailure || !['success', 'scene_preview'].includes(episode.status));
  $('stages').textContent = `${episode.stage} / ${episode.world.mission.length} OBJECTIVES`;
  $('scrub').max = episode.duration || 1; $('scrub').value = 0; $('duration').textContent = clock(episode.duration);
  $('record-status').textContent = episode.status === 'scene_preview' ? 'SCENE ONLY' : `${episode.contacts} CONTACTS · ${episode.violations} RULE VIOLATIONS`;
  $('loading').classList.add('hidden'); setView(episode.status === 'scene_preview' ? 'orbit' : 'chase');
  setPlaying(episode.status !== 'scene_preview');
}
function frame(now) {
  const dt = Math.min((now - previous) / 1000, .1); previous = now;
  if (episode) {
    if (playing) { playback = Math.min(episode.duration, playback + dt * Number($('rate').value)); if (playback === episode.duration) setPlaying(false); }
    while (active + 1 < episode.trace.length && episode.trace[active + 1].time <= playback) active++;
    while (active > 0 && episode.trace[active].time > playback) active--;
    const point = episode.trace[active], next = episode.trace[Math.min(active + 1, episode.trace.length - 1)];
    const fraction = next.time > point.time ? Math.max(0, Math.min(1, (playback - point.time) / (next.time - point.time))) : 0;
    view.draw(point, next, fraction, active, dt, playing);
    $('scrub').value = playback; $('elapsed').textContent = clock(playback);
    const completed = (episode.reports || []).filter(r => r.valid && r.time <= playback).length;
    $('stages').textContent = `${completed} / ${episode.world.mission.length} OBJECTIVES`;
    $('altitude').textContent = point.position[2].toFixed(2);
    $('speed').textContent = Math.hypot(...point.velocity.slice(0, 2)).toFixed(2);
    const room = Object.entries(episode.world.rooms).find(([, [lo, hi]]) => point.position[0] >= lo[0] && point.position[0] <= hi[0] && point.position[1] >= lo[1] && point.position[1] <= hi[1]);
    $('room').textContent = title(room?.[0] || 'Doorway');
    $('position').textContent = point.position.map((v, i) => `${'XYZ'[i]} ${v.toFixed(2)}`).join(' · ');
    const decision = episode.decisions.findLast(d => d.role === 'control' && d.time <= playback);
    $('decision').textContent = decision ? decision.choice.replaceAll('_', ' ') : 'Awaiting first decision';
    const intent = episode.intents?.findLast(d => d.time <= playback);
    $('phase').textContent = intent ? `${intent.phase.toUpperCase()} · JEV'S TASK` : 'MISSION INTENT';
    $('task').textContent = completed === episode.world.mission.length ? 'Mission complete' : intent?.task || 'Awaiting first task';
    $('survey').textContent = intent?.survey ? `${intent.survey} at last plan · occluded space remains unknown` : '';
    const done = episode.status !== 'scene_preview' && completed === episode.world.mission.length;
    const phase = done ? 'done' : intent?.phase === 'return' ? 'return' : ['search', 'approach', 'report'].includes(intent?.phase) ? 'search' : 'navigate';
    const order = ['navigate', 'search', 'return'];
    document.querySelectorAll('[data-phase]').forEach(el => {
      el.classList.toggle('active', el.dataset.phase === phase);
      el.classList.toggle('done', done || order.indexOf(el.dataset.phase) < order.indexOf(phase));
      if (el.dataset.phase === phase) el.setAttribute('aria-current', 'step'); else el.removeAttribute('aria-current');
    });
    const target = episode.world.mission[completed];
    const stateText = {
      navigate: `Heading to the ${target?.room || 'launch point'}`,
      search: intent?.phase === 'approach' ? `Moving closer to inspect ${target?.marker || 'the marker'}` : intent?.phase === 'report' ? 'Checking the objective' : `Looking for the ${target?.marker || ''} marker`,
      return: 'Returning to the starting point',
      done: episode.contacts > 0 || episode.violations > 0 ? 'Objectives complete · safety check failed' : 'Mission complete',
    };
    const latestControl = decision && (!intent || decision.time >= intent.time);
    const action = latestControl && decision.choice === 'brake_and_replan' ? 'Replanning the route'
      : latestControl && decision.choice.startsWith('detour_') ? 'Trying a way around an obstacle' : stateText[phase];
    $('now').textContent = episode.status === 'scene_preview' ? 'Scene preview · no flight playing' : playback >= episode.duration && !done ? 'Recording ended · mission unfinished' : done ? stateText.done : action;
  }
  requestAnimationFrame(frame);
}
$('play').addEventListener('click', () => { if (playback >= episode.duration) playback = 0; setPlaying(!playing); });
$('restart').addEventListener('click', () => { playback = 0; active = 0; view.snap = true; setPlaying(true); });
$('scrub').addEventListener('input', e => {
  const requested = Number(e.target.value);
  // Range inputs can round decimal timestamps just below the final report time.
  playback = Math.abs(requested - episode.duration) < 1e-6 ? episode.duration : requested;
  view.snap = true;
});
$('cutaway').addEventListener('change', e => view.setCutaway(e.target.checked));
$('trail').addEventListener('change', e => { view.trailVisible = e.target.checked; });
function showDetails(open) {
  $('details').hidden = !open;
  document.body.classList.toggle('details-open', open);
  $('focus').setAttribute('aria-expanded', String(open));
}
$('focus').addEventListener('click', () => showDetails($('details').hidden));
$('close-details').addEventListener('click', () => { showDetails(false); $('focus').focus(); });
window.addEventListener('keydown', e => { if (e.key === 'Escape' && !$('details').hidden) { showDetails(false); $('focus').focus(); } });
$('fullscreen').addEventListener('click', async () => { try { if (document.fullscreenElement) await document.exitFullscreen(); else await document.documentElement.requestFullscreen(); } catch { $('fullscreen').title = 'Fullscreen is unavailable in this browser panel'; } });
document.querySelectorAll('[data-view]').forEach(b => b.addEventListener('click', () => setView(b.dataset.view)));
view.controls.addEventListener('start', () => { if (view.mode !== 'orbit') setView('orbit'); });
$('run').addEventListener('change', e => load(e.target.value).catch(showError));
window.addEventListener('keydown', e => { if (e.code === 'Space' && !['INPUT', 'SELECT', 'BUTTON'].includes(document.activeElement.tagName)) { e.preventDefault(); setPlaying(!playing); } });
function showError(error) { $('loading').classList.remove('hidden'); $('loading').textContent = error.message; }
try {
  const response = await fetch('data/index.json', { cache: 'no-store' }); if (!response.ok) throw new Error('No exported flights. Run export_house_demo.py.');
  const runs = await response.json();
  $('run').replaceChildren(...runs.map(run => { const option = document.createElement('option'); option.value = run.path; option.textContent = run.label; return option; }));
  await load(runs[0].path);
} catch (error) { showError(error); }
requestAnimationFrame(frame);
