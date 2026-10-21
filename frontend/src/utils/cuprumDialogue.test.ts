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
  signalImagery,
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

function choicesFor(
  node: CuprumNode,
  ctx: CuprumContext = BASE_CONTEXT,
  runtime: CuprumRuntime = createCuprumRuntime(),
): CuprumChoice[] {
  return typeof node.choices === 'function' ? node.choices(ctx, runtime) : node.choices;
}

describe('cuprumDialogue', () => {
  it('gives the root exactly three choices and every node three replies', () => {
    const root = resolveCuprumNode(ROOT_NODE_ID, BASE_CONTEXT, createCuprumRuntime());
    expect(root).not.toBeNull();
    expect(root?.choices).toHaveLength(3);

    for (const node of Object.values(CUPRUM_NODES)) {
      expect(choicesFor(node)).toHaveLength(3);
    }
  });

  it('keeps every dialogue link inside the tree and reachable from the root', () => {
    for (const node of Object.values(CUPRUM_NODES)) {
      for (const choice of choicesFor(node)) {
        if (choice.close) {
          expect(choice.next).toBeUndefined();
        } else {
          expect(choice.next).toBeTruthy();
          expect(CUPRUM_NODES[choice.next as string]).toBeDefined();
        }
      }
    }

    const reachable = new Set<string>([ROOT_NODE_ID]);
    const queue = [ROOT_NODE_ID];
    while (queue.length > 0) {
      const id = queue.shift() as string;
      const node = CUPRUM_NODES[id];
      for (const choice of choicesFor(node)) {
        if (choice.next && !reachable.has(choice.next)) {
          reachable.add(choice.next);
          queue.push(choice.next);
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
    expect(next.storyIndex).toBe(1);
    expect(next.turn).toBe(1);
    expect(resolveCuprumNode('story', BASE_CONTEXT, next)?.speech).toBe(
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
      "What's the loose thread?",
    );

    const driver = resolveCuprumNode('driver-detail', ctx, runtime);
    expect(driver?.speech).toContain('far side of the world');
    expect(driver?.choices).toHaveLength(3);

    const change = resolveCuprumNode('change', ctx, { ...runtime, freshData: true });
    expect(change?.speech).toContain('wind changed');

    const memory = resolveCuprumNode('memory-why', ctx, runtime);
    expect(memory?.speech).toContain('September');
  });

  it('keeps every spoken node free of raw numbers and advice', () => {
    const runtime = createCuprumRuntime();
    for (const node of Object.values(CUPRUM_NODES)) {
      const resolved = resolveCuprumNode(node.id, BASE_CONTEXT, runtime);
      expect(resolved?.speech, node.id).not.toMatch(/\d|[$%]/);
      expect(resolved?.speech, node.id).not.toMatch(/\b(buy|sell|target|advice)\b/i);
    }
  });

  it('maps signal labels to concrete, non-numeric imagery', () => {
    expect(signalImagery(BASE_CONTEXT)).toBe('the paper weather');
    expect(
      signalImagery({ ...BASE_CONTEXT, signalLabel: 'China PMI', signalCategory: 'Macro' }),
    ).toBe('the far side of the world');
    expect(
      signalImagery({ ...BASE_CONTEXT, signalLabel: null, signalCategory: 'Momentum' }),
    ).toBe('the current');
    expect(
      signalImagery({ ...BASE_CONTEXT, signalLabel: null, signalCategory: null }),
    ).toBe('the hum');
  });
});
