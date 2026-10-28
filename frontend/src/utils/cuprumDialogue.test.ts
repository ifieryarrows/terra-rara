import { describe, expect, it } from 'vitest';
import type { CommentaryResponse } from '../types';
import {
  applyCuprumChoice,
  buildCuprumContext,
  createCuprumRuntime,
  CUPRUM_NODES,
  CUPRUM_STORIES,
  pickIdleLine,
  resolveCuprumNode,
  ROOT_NODE_ID,
  type CuprumChoice,
  type CuprumContext,
  type CuprumNode,
  type CuprumRuntime,
} from './cuprumDialogue';
import { buildCuprumBriefing } from './cuprumVoice';

const COMMENTARY: CommentaryResponse = {
  symbol: 'HG=F',
  commentary: 'Copper looks constructive.',
  error: null,
  generated_at: '2026-10-16T21:05:00Z',
  ai_stance: 'BEARISH',
  generation_mode: 'llm',
  sections: {
    synthesis: 'Copper looks constructive.',
    risks: ['A', 'B'],
    opportunities: ['C'],
    xai_brief: null,
    bias_warning: 'Macro week.',
  },
  companion_state: {
    character: 'Cuprum',
    mood: 'patina_bear',
    stance: 'BEARISH',
    energy: 'volatile',
    confidence_band: 'low',
    headline_quote: 'Quiet.',
    xai_highlights: {
      top_signals: [
        {
          feature: 'GXC_ret5',
          weight: 0.2,
          label: 'China ETF momentum',
          description: 'Chinese equity momentum has led copper demand expectations.',
          category: 'Momentum',
          time_horizon: '5d',
        },
      ],
      memory_anchors: [{ date: '2026-09-16', weight: 0.033 }],
    },
  },
};

const BASE_CONTEXT: CuprumContext = {
  mood: 'molten_bull',
  energy: 'charged',
  confidence: 'high',
  stance: 'BULLISH',
  signalCategory: 'Macro',
  signalLabel: 'Dollar index',
  riskCount: 2,
  opportunityCount: 1,
  hasBiasWarning: true,
  dataState: 'ready',
  updatedAt: '2026-10-16T21:05:00Z',
  briefing: buildCuprumBriefing(COMMENTARY, null, null),
};

/** A payload shape with no driver and no memory anchor: the sparse end of the range. */
const SPARSE_CONTEXT: CuprumContext = {
  ...BASE_CONTEXT,
  briefing: buildCuprumBriefing(null, null, null),
};

const CONTEXTS = [BASE_CONTEXT, SPARSE_CONTEXT];

const NEUTRAL_CONTEXT: CuprumContext = { ...BASE_CONTEXT, mood: 'tempered_neutral' };
const BEAR_CONTEXT: CuprumContext = { ...BASE_CONTEXT, mood: 'patina_bear' };
const MOOD_CONTEXTS: CuprumContext[] = [
  BASE_CONTEXT,
  SPARSE_CONTEXT,
  NEUTRAL_CONTEXT,
  BEAR_CONTEXT,
];

/** Measured against the real bubble: 220 chars fits the 18rem stage. */
const SPEECH_BUDGET = 220;
const ACTION_BUDGET = 90;

function choicesFor(
  node: CuprumNode,
  ctx: CuprumContext = BASE_CONTEXT,
  runtime: CuprumRuntime = createCuprumRuntime(),
): CuprumChoice[] {
  return typeof node.choices === 'function' ? node.choices(ctx, runtime) : node.choices;
}

function tellStories(runtime: CuprumRuntime, times: number): CuprumRuntime {
  let next = runtime;
  for (let index = 0; index < times; index += 1) {
    next = applyCuprumChoice(next, {
      id: 'story-installment',
      label: 'Tell me another.',
      next: 'story',
      advanceStory: true,
    });
  }
  return next;
}

describe('cuprumDialogue', () => {
  it('gives the root four choices and every other node three replies', () => {
    const root = resolveCuprumNode(ROOT_NODE_ID, BASE_CONTEXT, createCuprumRuntime());
    expect(root).not.toBeNull();
    expect(root?.choices).toHaveLength(4);

    for (const ctx of CONTEXTS) {
      for (const node of Object.values(CUPRUM_NODES)) {
        const expected = node.id === ROOT_NODE_ID ? 4 : 3;
        expect(choicesFor(node, ctx), `${node.id} @ ${ctx.dataState}`).toHaveLength(expected);
      }
    }
  });

  it('keeps every dialogue link inside the tree and reachable from the root', () => {
    for (const ctx of CONTEXTS) {
      for (const node of Object.values(CUPRUM_NODES)) {
        for (const choice of choicesFor(node, ctx)) {
          if (choice.close) {
            expect(choice.next).toBeUndefined();
          } else {
            expect(choice.next).toBeTruthy();
            expect(CUPRUM_NODES[choice.next as string]).toBeDefined();
          }
        }
      }
    }

    // Data-dependent choices can hide a branch, so reachability is the union
    // over every payload shape we support.
    const reachable = new Set<string>();
    for (const ctx of CONTEXTS) {
      const queue = [ROOT_NODE_ID];
      const seen = new Set<string>([ROOT_NODE_ID]);
      while (queue.length > 0) {
        const id = queue.shift() as string;
        const node = CUPRUM_NODES[id];
        for (const choice of choicesFor(node, ctx)) {
          if (!choice.next) continue;
          reachable.add(choice.next);
          if (!seen.has(choice.next)) {
            seen.add(choice.next);
            queue.push(choice.next);
          }
        }
      }
    }

    expect([...reachable].sort()).toEqual(Object.keys(CUPRUM_NODES).sort());
  });

  it('speaks in a different voice for each mood', () => {
    const runtime = createCuprumRuntime();
    const bull = resolveCuprumNode(ROOT_NODE_ID, BASE_CONTEXT, runtime);
    const bear = resolveCuprumNode(
      ROOT_NODE_ID,
      { ...BASE_CONTEXT, mood: 'patina_bear' },
      runtime,
    );
    const neutral = resolveCuprumNode(
      ROOT_NODE_ID,
      { ...BASE_CONTEXT, mood: 'tempered_neutral' },
      runtime,
    );

    expect(bull?.speech).not.toEqual(bear?.speech);
    expect(bear?.speech).not.toEqual(neutral?.speech);
    expect(bull?.speech).toContain('mid-roar');
    expect(neutral?.speech).toContain('space between prices');
  });

  it('acknowledges loading, error, empty and fresh snapshots without breaking character', () => {
    const runtime = createCuprumRuntime();
    expect(
      resolveCuprumNode(ROOT_NODE_ID, { ...BASE_CONTEXT, dataState: 'loading' }, runtime)?.speech,
    ).toContain('wires are still warm');
    expect(
      resolveCuprumNode(ROOT_NODE_ID, { ...BASE_CONTEXT, dataState: 'error' }, runtime)?.speech,
    ).toContain('lost the thread');
    expect(
      resolveCuprumNode(ROOT_NODE_ID, { ...BASE_CONTEXT, dataState: 'empty' }, runtime)?.speech,
    ).toContain('still here');
    expect(
      resolveCuprumNode(
        ROOT_NODE_ID,
        BASE_CONTEXT,
        createCuprumRuntime({ freshData: true }),
      )?.speech,
    ).toContain('wind changed');
  });

  it('warms up the greeting once the bond is high', () => {
    const speech = resolveCuprumNode(
      ROOT_NODE_ID,
      BASE_CONTEXT,
      createCuprumRuntime({ bond: 5 }),
    )?.speech;
    expect(speech).toContain('kept the forge warm');
  });

  it('turns risk and opportunity counts into weather, never numbers', () => {
    const knot = resolveCuprumNode(
      'sense',
      { ...BASE_CONTEXT, riskCount: 3, opportunityCount: 1 },
      createCuprumRuntime(),
    );
    const doors = resolveCuprumNode(
      'sense',
      { ...BASE_CONTEXT, riskCount: 1, opportunityCount: 3 },
      createCuprumRuntime(),
    );

    expect(knot?.speech).toContain('More knots than doors');
    expect(doors?.speech).toContain('doors are winning');
    expect(knot?.speech).not.toMatch(/\d/);
    expect(doors?.speech).not.toMatch(/\d/);
  });

  it('keeps low-confidence answers soft and high-confidence answers direct', () => {
    const low = resolveCuprumNode(
      'sense-worry',
      { ...BASE_CONTEXT, confidence: 'low' },
      createCuprumRuntime(),
    );
    const high = resolveCuprumNode(
      'sense-worry',
      { ...BASE_CONTEXT, confidence: 'high' },
      createCuprumRuntime(),
    );

    expect(low?.speech).toContain('whisper');
    expect(high?.speech).toContain('out loud');
  });

  it('advances the story loop and remembers the turn', () => {
    const start = createCuprumRuntime();
    const first = resolveCuprumNode('story', BASE_CONTEXT, start);
    expect(first?.speech).toBe(CUPRUM_STORIES[0].text);

    const storyChoice = choicesFor(CUPRUM_NODES.story).find((choice) => choice.advanceStory);
    expect(storyChoice).toBeDefined();
    const next = applyCuprumChoice(start, storyChoice!);
    expect(next.storiesTold).toBe(1);
    expect(next.turn).toBe(1);
    expect(resolveCuprumNode('story', BASE_CONTEXT, next)?.speech).toBe(
      CUPRUM_STORIES[0].text,
    );

    // Asking again moves to the next story instead of repeating the first.
    const after = applyCuprumChoice(next, storyChoice!);
    expect(after.storiesTold).toBe(2);
    expect(resolveCuprumNode('story', BASE_CONTEXT, after)?.speech).toBe(
      CUPRUM_STORIES[1].text,
    );
  });

  it('raises the bond on warm, curious or teasing choices and caps it', () => {
    const warm = applyCuprumChoice(createCuprumRuntime(), {
      id: 'x',
      label: 'x',
      effect: 'warm',
    });
    expect(warm.bond).toBe(1);

    const capped = applyCuprumChoice(createCuprumRuntime({ bond: 9 }), {
      id: 'x',
      label: 'x',
      effect: 'tease',
    });
    expect(capped.bond).toBe(9);
  });

  it('picks ambient lines that avoid the recent ones', () => {
    const first = pickIdleLine(
      BASE_CONTEXT,
      createCuprumRuntime(),
      [],
      () => 0,
    );
    const second = pickIdleLine(
      BASE_CONTEXT,
      createCuprumRuntime(),
      [first.id],
      () => 0,
    );

    expect(second.id).not.toBe(first.id);
  });

  it('prioritises a data-state line while the snapshot is loading', () => {
    const line = pickIdleLine(
      { ...BASE_CONTEXT, dataState: 'loading' },
      createCuprumRuntime(),
      [],
      () => 0,
    );
    expect(line.id.startsWith('loading-')).toBe(true);
  });

  it('derives mood, energy, confidence and model focus from the API contract', () => {
    const ctx = buildCuprumContext(COMMENTARY, null, null);
    expect(ctx.mood).toBe('patina_bear');
    expect(ctx.energy).toBe('volatile');
    expect(ctx.confidence).toBe('low');
    expect(ctx.riskCount).toBe(2);
    expect(ctx.opportunityCount).toBe(1);
    expect(ctx.hasBiasWarning).toBe(true);
    expect(ctx.signalCategory).toBe('Momentum');
    expect(ctx.signalLabel).toBe('China ETF momentum');
    expect(ctx.briefing.driver?.phrase).toBe('the far side of the world');
  });

  it('turns the XAI briefing into deeper questions and grounded answers', () => {
    const ctx: CuprumContext = {
      ...BASE_CONTEXT,
      briefing: buildCuprumBriefing(COMMENTARY, null, null),
    };
    const runtime = createCuprumRuntime();

    const sense = resolveCuprumNode('sense', ctx, runtime);
    expect(sense?.choices.map((choice) => choice.label)).toContain(
      "Who's doing the talking?",
    );

    const driver = resolveCuprumNode('driver-detail', ctx, runtime);
    expect(driver?.speech).toContain('far side of the world');
    expect(driver?.choices).toHaveLength(3);

    const change = resolveCuprumNode('change', ctx, { ...runtime, freshData: true });
    expect(change?.speech).toContain('wind changed');

    const memory = resolveCuprumNode('memory-why', ctx, runtime);
    expect(memory?.speech).toContain('September');
  });

  it('uses the fresh-snapshot greeting only at the start of a conversation', () => {
    const fresh = createCuprumRuntime({ freshData: true });

    expect(resolveCuprumNode(ROOT_NODE_ID, BASE_CONTEXT, fresh)?.speech).toContain(
      'wind changed',
    );

    const later = applyCuprumChoice(fresh, {
      id: 'ask-again',
      label: 'Let me ask something else.',
      next: ROOT_NODE_ID,
    });
    expect(resolveCuprumNode(ROOT_NODE_ID, BASE_CONTEXT, later)?.speech).not.toContain(
      'wind changed',
    );
  });

  it('welcomes return within conversation without repeating arrival greetings or "You came back"', () => {
    const runtime = applyCuprumChoice(createCuprumRuntime({ bond: 5 }), {
      id: 'ask-again',
      label: 'Let me ask something else.',
      next: ROOT_NODE_ID,
    });
    expect(runtime.turn).toBe(1);
    const resolved = resolveCuprumNode(ROOT_NODE_ID, BASE_CONTEXT, runtime);
    expect(resolved?.speech).not.toContain('You came back');
    expect(resolved?.speech).not.toContain('You caught me mid-roar');
    expect(resolved?.speech).toContain('Fire away. What else is on your mind?');

    const bear = resolveCuprumNode(ROOT_NODE_ID, { ...BASE_CONTEXT, mood: 'patina_bear' }, runtime);
    expect(bear?.speech).not.toContain('You came back');
    expect(bear?.speech).toContain("Go ahead, I'm still listening. What else?");

    const neutral = resolveCuprumNode(ROOT_NODE_ID, { ...BASE_CONTEXT, mood: 'tempered_neutral' }, runtime);
    expect(neutral?.speech).not.toContain('You came back');
    expect(neutral?.speech).toContain('Still here with you. What else did you want to check?');
  });

  it('lets the reader ask the console narrative in Cuprum\'s own words', () => {
    const ctx: CuprumContext = {
      ...BASE_CONTEXT,
      briefing: buildCuprumBriefing(COMMENTARY, null, null),
    };
    const runtime = createCuprumRuntime();

    const driver = resolveCuprumNode('driver-detail', ctx, runtime);
    expect(driver?.choices.map((choice) => choice.label)).toContain(
      'What does the board say?',
    );

    const read = resolveCuprumNode('board-read', ctx, runtime);
    expect(read?.speech).toMatch(/board reads like this/i);
    expect(read?.speech).not.toMatch(/\d|[$%]/);
    expect(read?.choices).toHaveLength(3);
  });

  it('keeps the fresh-snapshot flag alive so "what changed" can still answer', () => {
    const fresh = createCuprumRuntime({ freshData: true });
    const afterChoice = applyCuprumChoice(fresh, {
      id: 'root-sense',
      label: 'What do you sense out there?',
      next: 'sense',
    });

    expect(afterChoice.freshData).toBe(true);
    expect(resolveCuprumNode('change', BASE_CONTEXT, afterChoice)?.speech).toContain(
      'wind changed',
    );
  });

  it('does not offer data questions while the snapshot is still loading', () => {
    const loading: CuprumContext = { ...BASE_CONTEXT, dataState: 'loading' };
    const labels = resolveCuprumNode('sense', loading, createCuprumRuntime())?.choices.map(
      (choice) => choice.label,
    );

    expect(labels).not.toContain("What's the loose thread?");
    expect(labels).toContain('Let me ask something else.');
  });

  it('makes bearish Cuprum increasingly weary and overwhelmed by repeated story requests until falling asleep', () => {
    const bearCtx: CuprumContext = { ...BASE_CONTEXT, mood: 'patina_bear' };
    let runtime = createCuprumRuntime();

    // 1st story request
    runtime = applyCuprumChoice(runtime, { id: 'root-story', label: "Tell me something I don't know.", next: 'story' });
    let resolved = resolveCuprumNode('story', bearCtx, runtime);
    expect(resolved?.speech).toContain('first smiths');

    // 2nd story request
    runtime = applyCuprumChoice(runtime, { id: 'story-installment', label: 'Tell me another.', next: 'story', advanceStory: true });
    resolved = resolveCuprumNode('story', bearCtx, runtime);
    expect(resolved?.speech).toContain('Statue of Liberty');

    // 3rd story request (weary & sighing)
    runtime = applyCuprumChoice(runtime, { id: 'story-installment', label: 'Tell me another.', next: 'story', advanceStory: true });
    resolved = resolveCuprumNode('story', bearCtx, runtime);
    expect(resolved?.speech).toMatch(/deep, raspy sigh|Another one/i);

    // 4th story request
    runtime = applyCuprumChoice(runtime, { id: 'story-installment', label: 'Tell me another.', next: 'story', advanceStory: true });
    resolved = resolveCuprumNode('story', bearCtx, runtime);
    expect(resolved?.speech).toMatch(/don't quit|hammering/i);

    // 5th story request (overwhelmed & complaining)
    runtime = applyCuprumChoice(runtime, { id: 'story-installment', label: 'Tell me another.', next: 'story', advanceStory: true });
    resolved = resolveCuprumNode('story', bearCtx, runtime);
    expect(resolved?.speech).toMatch(/genuinely overwhelmed|sit in the quiet/i);

    // 6th story request (bunalmış / completely suffocated)
    runtime = applyCuprumChoice(runtime, { id: 'story-installment', label: 'Tell me another.', next: 'story', advanceStory: true });
    resolved = resolveCuprumNode('story', bearCtx, runtime);
    expect(resolved?.speech).toMatch(/completely suffocated|bunalmış|drained/i);

    // 7th story request (user exaggerated -> passes out into deep sleep)
    runtime = applyCuprumChoice(runtime, { id: 'story-installment', label: 'Tell me another.', next: 'story', advanceStory: true });
    resolved = resolveCuprumNode('story', bearCtx, runtime);
    expect(resolved?.speech).toMatch(/Zzz|exhausted sleep/i);
    expect(resolved?.action).toMatch(/fallen asleep|snores/i);
    expect(resolved?.choices.map((c) => c.label)).toContain('[Tuck him in] Rest well, little one.');

    // Poking him while asleep
    runtime = applyCuprumChoice(runtime, { id: 'bear-sleep-poke', label: '[Poke him gently] ...Cuprum?', next: 'story' });
    resolved = resolveCuprumNode('story', bearCtx, runtime);
    expect(resolved?.speech).toMatch(/bother the bull|off the clock/i);
  });

  it('keeps every spoken node free of raw numbers and advice (wires reports attention percentages without advice or price targets)', () => {
    const runtime = createCuprumRuntime();
    for (const node of Object.values(CUPRUM_NODES)) {
      const resolved = resolveCuprumNode(node.id, BASE_CONTEXT, runtime);
      if (node.id === 'wires') {
        expect(resolved?.speech, node.id).toMatch(/%/);
        expect(resolved?.speech, node.id).not.toMatch(/\$/);
      } else {
        expect(resolved?.speech, node.id).not.toMatch(/\d|[$%]/);
      }
      expect(resolved?.speech, node.id).not.toMatch(/\b(buy|sell|target|advice)\b/i);
    }
  });

  it('reports exact top-3 attention weight percentages on the wires node without advice or price targets', () => {
    const runtime = createCuprumRuntime();
    const bull = resolveCuprumNode('wires', BASE_CONTEXT, runtime);
    expect(bull?.speech).toContain('%');
    expect(bull?.speech).toContain('China ETF momentum');
    expect(bull?.speech).toMatch(/is eating 20\.0% of the fire/);
    expect(bull?.speech).not.toMatch(/\$|\b(buy|sell|target|price)\b/i);

    const bear = resolveCuprumNode('wires', { ...BASE_CONTEXT, mood: 'patina_bear' }, runtime);
    expect(bear?.speech).toMatch(/20\.0% of my headache is China ETF momentum/);

    const neutral = resolveCuprumNode('wires', { ...BASE_CONTEXT, mood: 'tempered_neutral' }, runtime);
    expect(neutral?.speech).toMatch(/China ETF momentum holds 20\.0% of the model's focus/);
  });

  it('keeps every spoken line and action inside the bubble budget', () => {
    const runtimes = [
      createCuprumRuntime(),
      tellStories(createCuprumRuntime(), 3),
      tellStories(createCuprumRuntime(), 8),
      tellStories(createCuprumRuntime(), 13),
    ];

    for (const ctx of MOOD_CONTEXTS) {
      for (const runtime of runtimes) {
        for (const node of Object.values(CUPRUM_NODES)) {
          const resolved = resolveCuprumNode(node.id, ctx, runtime);
          expect(
            (resolved?.speech ?? '').length,
            `${node.id} speech @ ${ctx.mood}`,
          ).toBeLessThanOrEqual(SPEECH_BUDGET);
          expect(
            (resolved?.action ?? '').length,
            `${node.id} action @ ${ctx.mood}`,
          ).toBeLessThanOrEqual(ACTION_BUDGET);
        }
      }
    }
  });

  it('runs out of stories in neutral and repeats the same line until the page reloads', () => {
    let runtime = applyCuprumChoice(createCuprumRuntime(), {
      id: 'root-story',
      label: "Tell me something I don't know.",
      next: 'story',
    });
    expect(resolveCuprumNode('story', NEUTRAL_CONTEXT, runtime)?.speech).toContain(
      'first smiths',
    );

    // All ten stories told.
    runtime = tellStories(runtime, CUPRUM_STORIES.length - 1);
    expect(resolveCuprumNode('story', NEUTRAL_CONTEXT, runtime)).not.toBeNull();

    runtime = tellStories(runtime, 1);
    const done = resolveCuprumNode('story', NEUTRAL_CONTEXT, runtime);
    expect(done?.speech).toMatch(/out of stories|no more stories|nothing left/i);
    expect(done?.choices.map((choice) => choice.label)).not.toContain('Tell me another.');
    // The story question stays available so asking again repeats the same answer.
    expect(done?.choices.map((choice) => choice.label)).toContain('Tell me a story.');

    runtime = tellStories(runtime, 1);
    expect(resolveCuprumNode('story', NEUTRAL_CONTEXT, runtime)?.speech).toBe(done?.speech);
    runtime = tellStories(runtime, 1);
    expect(resolveCuprumNode('story', NEUTRAL_CONTEXT, runtime)?.speech).toBe(done?.speech);
  });

  it('keeps the bull telling stories and heats up after the first two', () => {
    let runtime = applyCuprumChoice(createCuprumRuntime(), {
      id: 'root-story',
      label: "Tell me something I don't know.",
      next: 'story',
    });
    const first = resolveCuprumNode('story', BASE_CONTEXT, runtime);
    expect(first?.speech).toContain('first smiths');

    runtime = tellStories(runtime, 1);
    expect(resolveCuprumNode('story', BASE_CONTEXT, runtime)?.speech).toContain(
      'Statue of Liberty',
    );

    runtime = tellStories(runtime, 1);
    const heated = resolveCuprumNode('story', BASE_CONTEXT, runtime);
    expect(heated?.speech).not.toBe(first?.speech);
    expect(heated?.speech).not.toContain('first smiths');
    expect(heated?.action).toMatch(/ember|spark|roar|heat|fire/i);
    expect(heated?.action).not.toBe(first?.action);

    // The bull never runs out; it just stays hot.
    runtime = tellStories(runtime, 6);
    const later = resolveCuprumNode('story', BASE_CONTEXT, runtime);
    expect((later?.speech ?? '').length).toBeGreaterThan(0);
    expect(later?.choices.map((choice) => choice.label)).toContain('Tell me another.');
  });

  it('offers the driver cast and its persistence question once the data is ready', () => {
    const sense = choicesFor(CUPRUM_NODES.sense);
    expect(sense.map((choice) => choice.next)).toContain('drivers');

    const cast = resolveCuprumNode('drivers', BASE_CONTEXT, createCuprumRuntime());
    const lead = BASE_CONTEXT.briefing.drivers[0];
    expect(lead).toBeTruthy();
    expect(cast?.speech.toLowerCase()).toContain(lead.phrase);
    expect(cast?.choices.map((choice) => choice.next)).toContain('driver-outlook');

    const outlook = resolveCuprumNode('driver-outlook', BASE_CONTEXT, createCuprumRuntime());
    expect(outlook?.speech).toMatch(/still/);
    expect(outlook?.choices).toHaveLength(3);
  });
});
