// @vitest-environment jsdom
import { describe, expect, it } from 'vitest';
import {
  advanceCinematicEntry,
  CINEMATIC_ENTRY_RESET,
  CINEMATIC_ENTRY_TRIGGER,
} from '../features/landing/cinematic-entry';
import {
  advanceLogoTransition,
  captureOutgoingBackdrop,
  INITIAL_LOGO_TRANSITION_STATE,
} from '../components/ui/LogoTransition';
import { createCuprumRuntime, resolveBearExhaustionStage } from './cuprumDialogue';
import { advanceSpeechBubble, createInitialSpeechBubbleState } from '../components/CuprumSpeechBubble';
import { advanceCuprumSession, INITIAL_CUPRUM_SESSION_STATE } from '../components/CuprumCompanionPanel';
import { advanceHeatmapPopover, INITIAL_HEATMAP_POPOVER_STATE } from '../features/heatmap/HeatmapPanel';
import { resolveLogoLoadPhase } from '../features/heatmap/CompanyLogo';
import { advanceDeferredPanels, INITIAL_DEFERRED_PANELS_STATE } from '../pages/OverviewPage';
import type { CategoryAnchor } from '../features/heatmap/HeatmapTreemap';
import type { HeatmapData } from '../features/heatmap/heatmap-layout';

const SAMPLE_LEAF_A: HeatmapData = {
  id: 'FCX',
  name: 'Freeport-McMoRan',
  changePercent: 2.1,
  price: 45.2,
  weight: 100,
};

const SAMPLE_RECT = { left: 10, top: 20, right: 110, bottom: 80, width: 100, height: 60 };

const SAMPLE_ANCHOR_A: CategoryAnchor = {
  id: 'cat-producers',
  name: 'Copper Producers',
  depth: 1,
  rect: SAMPLE_RECT,
  containerRect: SAMPLE_RECT,
};

describe('Frontend explicit state machines', () => {
  describe('Cinematic entry hysteresis machine', () => {
    it('stays idle until trigger threshold and stays revealed inside hysteresis deadzone', () => {
      expect(advanceCinematicEntry('idle', 0.90)).toEqual({
        phase: 'idle',
        changed: false,
        target: 0,
      });
      expect(advanceCinematicEntry('idle', CINEMATIC_ENTRY_TRIGGER)).toEqual({
        phase: 'revealed',
        changed: true,
        target: 1,
      });
      expect(advanceCinematicEntry('revealed', 0.90)).toEqual({
        phase: 'revealed',
        changed: false,
        target: 1,
      });
      expect(advanceCinematicEntry('revealed', CINEMATIC_ENTRY_RESET - 0.01)).toEqual({
        phase: 'idle',
        changed: true,
        target: 0,
      });
    });
  });

  describe('LogoTransition state machine', () => {
    it('transitions through scatter, assemble, hold, release, reveal in full motion, or hold directly in reduced motion', () => {
      const origin = { left: 120, top: 80, width: 48, height: 48 };
      const started = advanceLogoTransition(INITIAL_LOGO_TRANSITION_STATE, {
        type: 'start',
        origin,
        reducedMotion: false,
      });
      expect(started).toEqual({
        active: true,
        phase: 'scatter',
        reducedMotion: false,
        origin,
      });

      const assembled = advanceLogoTransition(started, { type: 'advance', phase: 'assemble' });
      expect(assembled.phase).toBe('assemble');

      const completed = advanceLogoTransition(assembled, { type: 'reset' });
      expect(completed).toEqual(INITIAL_LOGO_TRANSITION_STATE);

      const reduced = advanceLogoTransition(INITIAL_LOGO_TRANSITION_STATE, {
        type: 'start',
        origin,
        reducedMotion: true,
      });
      expect(reduced.phase).toBe('hold');
    });

    it('captures the outgoing landing atmosphere and dispatches particle snapshot before navigation', () => {
      document.body.innerHTML = `
        <div class="cm-landing">
          <div class="cm-global-atmosphere cm-terra-atmosphere--landing-cinematic">
            <div class="cm-atmosphere-base" style="background-color: rgb(0, 0, 0);"></div>
            <div class="cm-atmosphere-glow cm-atmosphere-glow--copper" style="opacity: 0.03;"></div>
            <div class="cm-atmosphere-glow cm-atmosphere-glow--blue" style="opacity: 0.03;"></div>
            <div class="cm-atmosphere-band" style="opacity: 0;"></div>
            <div class="cm-atmosphere-void" style="background-color: rgb(0, 0, 0); opacity: 1;"></div>
            <div class="cm-atmosphere-stars" style="opacity: 0.56;"></div>
          </div>
          <canvas class="cm-particle-world" data-renderer="webgl2"></canvas>
        </div>
        <div id="snap-host" class="cm-logo-transition__snapshot">
          <div class="cm-global-atmosphere cm-logo-transition__snapshot-atmo">
            <div class="cm-atmosphere-base"></div>
            <div class="cm-atmosphere-glow cm-atmosphere-glow--copper"></div>
            <div class="cm-atmosphere-glow cm-atmosphere-glow--blue"></div>
            <div class="cm-atmosphere-band"></div>
            <div class="cm-atmosphere-void"></div>
            <div class="cm-atmosphere-stars"><canvas class="cm-atmosphere-starfield"></canvas></div>
            <div class="cm-atmosphere-grain"></div>
          </div>
          <canvas class="cm-logo-transition__snapshot-particles"></canvas>
        </div>
      `;

      let snapshotRequested = false;
      const onSnapshot = () => {
        snapshotRequested = true;
      };
      window.addEventListener('cm:particle-snapshot', onSnapshot);

      const host = document.getElementById('snap-host') as HTMLDivElement;
      captureOutgoingBackdrop(host);

      window.removeEventListener('cm:particle-snapshot', onSnapshot);

      expect(snapshotRequested).toBe(true);
      expect(host.dataset.captured).toBe('true');
      expect(host.style.backgroundColor).toBe('rgb(0, 0, 0)');
      const snapshotAtmo = host.querySelector('.cm-logo-transition__snapshot-atmo');
      expect(snapshotAtmo?.classList.contains('cm-terra-atmosphere--landing-cinematic')).toBe(true);
      expect(
        (host.querySelector('.cm-atmosphere-void') as HTMLElement | null)?.style.opacity,
      ).toBe('1');

      // When GalaxyTransitionField has absorbed live galaxy particles, snapshot omits duplicate static galaxy
      snapshotRequested = false;
      delete host.dataset.captured;
      document.documentElement.dataset.cmGalaxyActive = 'true';
      window.addEventListener('cm:particle-snapshot', onSnapshot);
      captureOutgoingBackdrop(host);
      window.removeEventListener('cm:particle-snapshot', onSnapshot);
      delete document.documentElement.dataset.cmGalaxyActive;

      expect(snapshotRequested).toBe(false);
      expect(host.dataset.captured).toBe('true');
      document.body.innerHTML = '';
    });
  });

  describe('Cuprum bear exhaustion state machine', () => {
    it('maps storyStreak and lastChoiceId to deterministic exhaustion stages', () => {
      expect(resolveBearExhaustionStage(createCuprumRuntime({ storyStreak: 0 }))).toBe('normal');
      expect(resolveBearExhaustionStage(createCuprumRuntime({ storyStreak: 2 }))).toBe('normal');
      expect(resolveBearExhaustionStage(createCuprumRuntime({ storyStreak: 3 }))).toBe('reluctant');
      expect(resolveBearExhaustionStage(createCuprumRuntime({ storyStreak: 4 }))).toBe('irritated');
      expect(resolveBearExhaustionStage(createCuprumRuntime({ storyStreak: 5 }))).toBe('overwhelmed');
      expect(resolveBearExhaustionStage(createCuprumRuntime({ storyStreak: 6 }))).toBe('drained');
      expect(resolveBearExhaustionStage(createCuprumRuntime({ storyStreak: 7 }))).toBe('asleep');
      expect(
        resolveBearExhaustionStage(
          createCuprumRuntime({ storyStreak: 7, lastChoiceId: 'bear-sleep-poke' }),
        ),
      ).toBe('asleep_poked');
    });
  });

  describe('SpeechBubble typewriter state machine', () => {
    it('transitions between typing, paused, and revealed states', () => {
      let state = createInitialSpeechBubbleState(6, false);
      expect(state.phase).toBe('typing');
      expect(state.visibleCount).toBe(0);

      state = advanceSpeechBubble(state, { type: 'tick', length: 6 });
      expect(state.visibleCount).toBe(1);
      expect(state.phase).toBe('typing');

      state = advanceSpeechBubble(state, { type: 'reveal-all', length: 6 });
      expect(state.phase).toBe('revealed');
      expect(state.visibleCount).toBe(6);

      state = advanceSpeechBubble(state, { type: 'set-paused', paused: true });
      expect(state.phase).toBe('paused');
      expect(state.paused).toBe(true);

      state = advanceSpeechBubble(state, { type: 'set-paused', paused: false });
      expect(state.phase).toBe('revealed');
      expect(state.paused).toBe(false);
    });
  });

  describe('CuprumCompanionPanel session state machine', () => {
    it('handles idle, dialogue navigation, typing, and close transitions', () => {
      const idleLine = { id: 'idle-1', text: 'Warm wires tonight.' };
      let state = advanceCuprumSession(INITIAL_CUPRUM_SESSION_STATE, {
        type: 'show-idle',
        line: idleLine,
      });
      expect(state.mode).toBe('idle');
      expect(state.idleLine).toEqual(idleLine);

      state = advanceCuprumSession(state, { type: 'set-typing', typing: true });
      expect(state.isTyping).toBe(true);

      state = advanceCuprumSession(state, { type: 'open-dialogue', nodeId: 'root' });
      expect(state.mode).toBe('dialogue');
      expect(state.activeNodeId).toBe('root');
      expect(state.idleLine).toBeNull();
      expect(state.isTyping).toBe(false);

      state = advanceCuprumSession(state, { type: 'advance-dialogue', nodeId: 'forecast' });
      expect(state.activeNodeId).toBe('forecast');

      state = advanceCuprumSession(state, { type: 'close-dialogue' });
      expect(state).toEqual(INITIAL_CUPRUM_SESSION_STATE);
    });
  });

  describe('HeatmapPopover interaction state machine', () => {
    it('arbitrates hover vs pin and toggles off when clicking the same pinned category', () => {
      let state = advanceHeatmapPopover(INITIAL_HEATMAP_POPOVER_STATE, {
        type: 'hover-open',
        anchor: SAMPLE_ANCHOR_A,
      });
      expect(state.mode).toBe('hovered');
      expect(state.anchor?.id).toBe('cat-producers');

      state = advanceHeatmapPopover(state, {
        type: 'leaf-hover',
        leaf: SAMPLE_LEAF_A,
      });
      expect(state.leaf?.id).toBe('FCX');

      state = advanceHeatmapPopover(state, {
        type: 'click-anchor',
        anchor: SAMPLE_ANCHOR_A,
      });
      expect(state.mode).toBe('pinned');
      expect(state.anchor?.id).toBe('cat-producers');

      // Hover close while pinned is ignored
      state = advanceHeatmapPopover(state, { type: 'hover-close' });
      expect(state.mode).toBe('pinned');

      // Clicking the same pinned category anchor unpins and returns to idle
      state = advanceHeatmapPopover(state, {
        type: 'click-anchor',
        anchor: SAMPLE_ANCHOR_A,
      });
      expect(state).toEqual(INITIAL_HEATMAP_POPOVER_STATE);
    });
  });

  describe('CompanyLogo & OverviewPage deferred panel machines', () => {
    it('resolves logo load phase deterministically', () => {
      expect(resolveLogoLoadPhase(null, false, null, null)).toBe('failed');
      expect(resolveLogoLoadPhase('/logos/FCX.png', false, null, null)).toBe('deferred');
      expect(resolveLogoLoadPhase('/logos/FCX.png', true, null, null)).toBe('loading');
      expect(resolveLogoLoadPhase('/logos/FCX.png', true, '/logos/FCX.png', null)).toBe('loaded');
      expect(resolveLogoLoadPhase('/logos/FCX.png', true, '/logos/FCX.png', '/logos/FCX.png')).toBe('failed');
    });

    it('transitions deferred panels monotonically to revealed', () => {
      const next = advanceDeferredPanels(INITIAL_DEFERRED_PANELS_STATE, {
        type: 'reveal',
        key: 'map',
      });
      expect(next).toEqual({
        companion: false,
        news: false,
        map: true,
      });
      expect(advanceDeferredPanels(next, { type: 'reveal', key: 'map' })).toBe(next);

      const all = advanceDeferredPanels(next, { type: 'reveal-all' });
      expect(all).toEqual({
        companion: true,
        news: true,
        map: true,
      });
      expect(advanceDeferredPanels(all, { type: 'reveal-all' })).toBe(all);
    });
  });
});
