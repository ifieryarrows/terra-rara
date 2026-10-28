import { describe, expect, it } from 'vitest';
import type {
  CommentaryResponse,
  ExplainFeatureEntry,
  ExplainResponse,
  TFTAnalysisResponse,
} from '../types';
import {
  buildCuprumBriefing,
  composeCuprumReply,
  signalPhrase,
  type CuprumIntent,
  type CuprumVoiceContext,
  type CuprumVoiceRuntime,
} from './cuprumVoice';

const COMMENTARY: CommentaryResponse = {
  symbol: 'HG=F',
  commentary: 'Copper looks constructive near 5.10.',
  error: null,
  generated_at: '2026-10-16T21:05:00Z',
  ai_stance: 'BULLISH',
  generation_mode: 'llm',
  sections: {
    synthesis:
      'Copper futures maintain a constructive five-day posture supported by tight spreads.',
    risks: [
      'Dollar rebound could cap upside momentum.',
      'Smelter restocking remains uneven.',
    ],
    opportunities: ['Warehouse draws tighten prompt COMEX spreads.'],
    xai_brief: 'The dollar is the largest encoder signal at 31%.',
    bias_warning: 'Subject to macro volatility around upcoming PMI releases.',
  },
  companion_state: {
    character: 'Cuprum',
    mood: 'molten_bull',
    stance: 'BULLISH',
    energy: 'charged',
    confidence_band: 'high',
    headline_quote: 'Copper looks constructive near 5.10.',
    xai_highlights: {
      horizon_label: 'five-day',
      top_signals: [
        {
          feature: 'DXY_level',
          weight: 0.31,
          label: 'Dollar index',
          description: 'A stronger dollar has historically capped copper upside.',
          category: 'Macro',
          time_horizon: '5d',
          recent: {
            basis: '5-session',
            kind: 'pct',
            last_value: 0.2,
            change: -0.42,
            change_pct: -0.4,
            as_of: '2026-10-16',
          },
        },
        {
          feature: 'GXC_ret5',
          weight: 0.2,
          label: 'China ETF momentum',
          description: 'Chinese equity momentum has led copper demand expectations.',
          category: 'Momentum',
          time_horizon: '5d',
          recent: {
            basis: '5-session',
            kind: 'pct',
            last_value: 0.11,
            change: 0.03,
            change_pct: 3.2,
            as_of: '2026-10-16',
          },
        },
      ],
      memory_anchors: [
        { date: '2026-09-16', weight: 0.033 },
        { date: '2026-08-14', weight: 0.018 },
      ],
    },
  },
};

const TFT = {
  direction: 'BEARISH',
  weekly_trend: 'BEARISH',
  primary_horizon: '5d',
  risk_level: 'HIGH',
  prediction: {
    baseline_staleness_days: 1,
    anomaly_detected: true,
    reference_price_date: '2026-10-16',
  },
} as TFTAnalysisResponse;

const EMPTY_BRIEFING = buildCuprumBriefing(null, null, null);

const RUNTIME: CuprumVoiceRuntime = { bond: 0, turn: 0, freshData: false };

function makeContext(overrides: Partial<CuprumVoiceContext> = {}): CuprumVoiceContext {
  return {
    mood: 'molten_bull',
    energy: 'charged',
    confidence: 'high',
    dataState: 'ready',
    riskCount: 2,
    opportunityCount: 1,
    briefing: buildCuprumBriefing(COMMENTARY, null, TFT),
    ...overrides,
  };
}

const INTENTS: CuprumIntent[] = [
  'spark',
  'sense',
  'sense-more',
  'sense-worry',
  'weight',
  'weight-change',
  'change',
  'driver-detail',
  'drivers',
  'driver-outlook',
  'memory-why',
  'risk',
  'opportunity',
  'board-read',
  'wires',
];

/** Measured against the real bubble: 220 chars fits the 18rem stage. */
const SPEECH_BUDGET = 220;

describe('cuprumVoice', () => {
  it('turns the old XAI payload into a safe, human briefing', () => {
    const briefing = buildCuprumBriefing(COMMENTARY, null, TFT);

    expect(briefing.driver?.phrase).toBe('the paper weather');
    expect(briefing.driver?.direction).toBe('cooling');
    expect(briefing.memory?.month).toBe('September');
    expect(briefing.risks[0]).toContain('paper weather');
    expect(briefing.risks[1]).toContain('warehouses');
    expect(briefing.opportunities[0]).toContain('spreads');
    expect(briefing.freshness).toBe('fresh');
    expect(briefing.anomaly).toBe(true);
    expect(briefing.biasWarning).toBe(true);
  });

  it('uses the narrative XAI brief when structured signals are missing', () => {
    const briefing = buildCuprumBriefing(
      {
        ...COMMENTARY,
        companion_state: {
          ...COMMENTARY.companion_state!,
          xai_highlights: null,
        },
      },
      null,
      null,
    );

    expect(briefing.driver?.phrase).toBe('the paper weather');
  });

  it('falls back to the explain payload when companion highlights are absent', () => {
    const explain = {
      variable_selection: {
        encoder: [
          {
            feature: 'Warehouse_draw',
            weight: 0.24,
            label: 'Warehouse draws',
            description: 'Prompt warehouse draws have tightened nearby spreads.',
            category: 'Sector',
            time_horizon: '5d',
          },
        ],
      },
      attention: {
        top_days: [{ date: '2026-08-14', weight: 0.18 }],
      },
      horizon: { label: 'five-day', days: 5 },
    } as unknown as ExplainResponse;

    const briefing = buildCuprumBriefing(
      {
        ...COMMENTARY,
        companion_state: {
          ...COMMENTARY.companion_state!,
          xai_highlights: null,
        },
      },
      explain,
      null,
    );

    expect(briefing.driver?.phrase).toBe('the warehouses');
    expect(briefing.memory?.month).toBe('August');
    expect(briefing.horizon).toBe('five-day');
  });

  it('reads the driver direction from a relative move, not from raw units', () => {
    const withRecent = (recent: NonNullable<ExplainFeatureEntry['recent']>) =>
      buildCuprumBriefing(
        {
          ...COMMENTARY,
          companion_state: {
            ...COMMENTARY.companion_state!,
            xai_highlights: {
              ...COMMENTARY.companion_state!.xai_highlights!,
              top_signals: [
                {
                  ...COMMENTARY.companion_state!.xai_highlights!.top_signals![0],
                  recent,
                },
              ],
            },
          },
        },
        null,
        null,
      );

    // 0.0002 on a base of 100 is 0.0002%: that is not a move worth calling.
    expect(
      withRecent({
        basis: '5-session',
        kind: 'abs',
        last_value: 100,
        change: 0.0002,
        change_pct: null,
        as_of: '2026-10-16',
      }).driver?.direction,
    ).toBe('steady');

    // 0.05 on a base of 0.45 is a double-digit percent move.
    expect(
      withRecent({
        basis: '5-session',
        kind: 'abs',
        last_value: 0.45,
        change: 0.05,
        change_pct: null,
        as_of: '2026-10-16',
      }).driver?.direction,
    ).toBe('warming');

    // A zero base cannot be turned into a percentage.
    expect(
      withRecent({
        basis: '5-session',
        kind: 'pct',
        last_value: 0,
        change: 0.4,
        change_pct: null,
        as_of: '2026-10-16',
      }).driver?.direction,
    ).toBe('unknown');
  });

  it('maps signal labels to concrete, non-numeric imagery', () => {
    expect(signalPhrase('Dollar index', 'Macro')).toBe('the paper weather');
    expect(signalPhrase('China PMI', 'Macro')).toBe('the far side of the world');
    expect(signalPhrase(null, 'Momentum')).toBe('the current');
    expect(signalPhrase(null, null)).toBe('the hum');
    expect(signalPhrase('Warehouse draws', 'Sector', 'Warehouse_draw')).toBe(
      'the warehouses',
    );
  });

  it('carries the whole console panel into the briefing', () => {
    const briefing = buildCuprumBriefing(COMMENTARY, null, TFT);

    expect(briefing.horizonPhrase).toBe('the next five sessions');
    expect(briefing.secondary?.phrase).toBe('the far side of the world');
    expect(briefing.secondary?.direction).toBe('warming');
    expect(briefing.memories).toEqual(['September', 'August']);
    expect(briefing.synthesis).toBe('the spreads are holding their breath');
  });

  it('talks about the horizon in its own words', () => {
    const speech = composeCuprumReply('sense-more', makeContext(), RUNTIME);

    expect(speech).toContain('the next five sessions');
    expect(speech).not.toMatch(/\d/);
  });

  it('names the second loudest signal', () => {
    const speech = composeCuprumReply('driver-detail', makeContext(), RUNTIME);

    expect(speech).toContain('the far side of the world');
    expect(speech).not.toMatch(/\d/);
  });

  it('remembers more than one attention day', () => {
    const speech = composeCuprumReply('memory-why', makeContext(), RUNTIME);

    expect(speech).toContain('September');
    expect(speech).toContain('August');
    expect(speech).not.toMatch(/\d/);
  });

  it('reads the board narrative without numbers', () => {
    const speech = composeCuprumReply('board-read', makeContext(), RUNTIME);

    expect(speech).toMatch(/board reads like this/i);
    expect(speech).toContain('the spreads are holding their breath');
    expect(speech).not.toMatch(/\d|[$%]/);
  });

  it('delivers exact driver weight percentages and real labels for wires intent', () => {
    const bullSpeech = composeCuprumReply('wires', makeContext({ mood: 'molten_bull' }), RUNTIME);
    expect(bullSpeech).toContain('Dollar index is eating 31.0% of the fire');
    expect(bullSpeech).toContain('China ETF momentum claims 20.0%');
    expect(bullSpeech).not.toMatch(/\$|\b(buy|sell|target)\b/i);

    const bearSpeech = composeCuprumReply('wires', makeContext({ mood: 'patina_bear' }), RUNTIME);
    expect(bearSpeech).toContain('31.0% of my headache is Dollar index');
    expect(bearSpeech).toContain('Another 20.0% is China ETF momentum');

    const neutralSpeech = composeCuprumReply('wires', makeContext({ mood: 'tempered_neutral' }), RUNTIME);
    expect(neutralSpeech).toContain("Dollar index holds 31.0% of the model's focus");
    expect(neutralSpeech).toContain('China ETF momentum follows at 20.0%');
  });

  it('grounds a market answer in the briefing without leaking raw numbers', () => {
    const speech = composeCuprumReply('sense', makeContext(), RUNTIME);

    expect(speech).toContain('paper weather');
    expect(speech).toMatch(/More knots than doors|warehouses|loose thread/);
    expect(speech).not.toMatch(/\d/);
    expect(speech).not.toMatch(/[$%]/);
  });

  it('keeps the mood in the wording while the facts stay the same', () => {
    const bull = composeCuprumReply('sense', makeContext(), RUNTIME);
    const bear = composeCuprumReply(
      'sense',
      makeContext({ mood: 'patina_bear', energy: 'volatile' }),
      RUNTIME,
    );

    expect(bull).not.toBe(bear);
    expect(bear).toContain('breathing slow');
  });

  it('answers what changed with the fresh-data flag and the live driver', () => {
    const speech = composeCuprumReply(
      'change',
      makeContext(),
      { ...RUNTIME, freshData: true },
    );

    expect(speech).toContain('wind changed');
    expect(speech).toContain('paper weather');
  });

  it('uses the old attention anchor as a memory, not a number', () => {
    const speech = composeCuprumReply('memory-why', makeContext(), RUNTIME);

    expect(speech).toContain('September');
    expect(speech).not.toMatch(/\d/);
  });

  it('stays in character while the wires are loading, empty or broken', () => {
    for (const dataState of ['loading', 'empty', 'error'] as const) {
      const speech = composeCuprumReply(
        'sense',
        makeContext({ dataState, briefing: EMPTY_BRIEFING }),
        RUNTIME,
      );
      expect(speech.length).toBeGreaterThan(20);
      expect(speech).not.toMatch(/\d/);
    }
  });

  it('does not invent a driver when the briefing has none', () => {
    const speech = composeCuprumReply(
      'sense',
      makeContext({ briefing: EMPTY_BRIEFING, riskCount: 0, opportunityCount: 0 }),
      RUNTIME,
    );

    expect(speech).toContain('hum');
    expect(speech).not.toMatch(/\d/);
  });

  it('filters numbers and advice out of every spoken intent', () => {
    const risky = buildCuprumBriefing(
      {
        ...COMMENTARY,
        sections: {
          synthesis: 'Buy now before the $5.10 target.',
          risks: ['Sell 12% at 6.20 before the drop.', 'Guaranteed $4.80 floor.'],
          opportunities: ['Target 7.40 for a 31% gain.'],
          xai_brief: 'DXY weight 31%.',
          bias_warning: 'Financial advice. Buy now.',
        },
        companion_state: {
          ...COMMENTARY.companion_state!,
          headline_quote: 'Buy at 5.10 for a 12% gain.',
        },
      },
      null,
      TFT,
    );

    for (const intent of INTENTS) {
      const speech = composeCuprumReply(
        intent,
        makeContext({ briefing: risky, riskCount: 2, opportunityCount: 1 }),
        RUNTIME,
      );
      if (intent === 'wires') {
        expect(speech, intent).toMatch(/%/);
        expect(speech, intent).not.toMatch(/\$/);
      } else {
        expect(speech, intent).not.toMatch(/\d/);
        expect(speech, intent).not.toMatch(/[$%]/);
      }
      expect(speech, intent).not.toMatch(/\b(buy|sell|target|advice)\b/i);
    }
  });

  it('is deterministic for the same turn', () => {
    const context = makeContext();
    expect(composeCuprumReply('sense', context, RUNTIME)).toBe(
      composeCuprumReply('sense', context, RUNTIME),
    );
  });

  it('keeps every composed reply inside the speech budget', () => {
    for (const mood of ['molten_bull', 'patina_bear', 'tempered_neutral'] as const) {
      for (const intent of INTENTS) {
        const speech = composeCuprumReply(intent, makeContext({ mood }), RUNTIME);
        expect(speech.length, `${intent} @ ${mood}`).toBeLessThanOrEqual(SPEECH_BUDGET);
      }
    }
  });
});

describe('cuprumVoice driver cast', () => {
  it('turns every encoder signal into a driver the pet can name', () => {
    const briefing = buildCuprumBriefing(COMMENTARY, null, TFT);

    expect(briefing.drivers).toHaveLength(2);
    expect(briefing.drivers[0]).toMatchObject({
      phrase: 'the paper weather',
      loudness: 'loud',
      share: 'three in ten',
      direction: 'cooling',
    });
    expect(briefing.drivers[1]).toMatchObject({
      phrase: 'the far side of the world',
      share: 'two in ten',
    });
  });

  it('speaks the whole cast instead of only the loudest one', () => {
    const speech = composeCuprumReply('drivers', makeContext(), RUNTIME);

    expect(speech).toContain('paper weather');
    expect(speech).toContain('far side of the world');
    expect(speech).toMatch(/loud/i);
    expect(speech).not.toMatch(/\d/);
  });

  it('answers "will it hold" with the direction it can see, never a promise', () => {
    const speech = composeCuprumReply('driver-outlook', makeContext(), RUNTIME);

    expect(speech).toMatch(/still/);
    expect(speech).toMatch(/not a promise|the direction shows/i);
    expect(speech).not.toMatch(/\d/);
  });
});
