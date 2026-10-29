import type {
  CommentaryResponse,
  ExplainFeatureEntry,
  ExplainRecent,
  ExplainResponse,
  TFTAnalysisResponse,
} from '../types';

export type CuprumMood = 'molten_bull' | 'patina_bear' | 'tempered_neutral';
export type CuprumEnergy = 'calm' | 'charged' | 'volatile';
export type CuprumConfidence = 'high' | 'medium' | 'low';
export type CuprumDataState = 'ready' | 'loading' | 'empty' | 'error';
export type CuprumDirection = 'warming' | 'cooling' | 'steady' | 'unknown';
export type CuprumFreshness = 'fresh' | 'stale' | 'unknown';

export interface CuprumDriverBriefing {
  /** Safe imagery used in speech, e.g. "the paper weather". */
  phrase: string;
  direction: CuprumDirection;
  /** A safe, non-numeric character note derived from the XAI description. */
  trait: string;
  horizon: string | null;
}

export interface CuprumMemoryBriefing {
  /** Month name only. The exact date never reaches the speech layer. */
  month: string;
  phrase: string;
}

/** One input the model is listening to, in the pet's own language. */
export interface CuprumDriverProfile {
  phrase: string;
  label: string;
  feature: string;
  weightPct: number;
  /** How much of the model's attention this input holds. */
  loudness: 'loud' | 'middling' | 'quiet';
  /** The same attention share said in words: "three in ten". */
  share: string;
  direction: CuprumDirection;
  horizon: string | null;
}

export interface CuprumBriefing {
  driver: CuprumDriverBriefing | null;
  /** The next loudest encoder signal, when the console sent more than one. */
  secondary: CuprumDriverBriefing | null;
  /** Every driver the console ranked, loudest first, so the pet can name them. */
  drivers: CuprumDriverProfile[];
  memory: CuprumMemoryBriefing | null;
  /** Every attention day the console highlighted, newest first. */
  memories: string[];
  risks: string[];
  opportunities: string[];
  /** The console narrative, told in Cuprum's language. */
  synthesis: string | null;
  biasWarning: boolean;
  freshness: CuprumFreshness;
  anomaly: boolean;
  horizon: string | null;
  /** The horizon in words, e.g. "the next five sessions". */
  horizonPhrase: string | null;
}

/** The slice of dialogue state the local voice needs. */
export interface CuprumVoiceContext {
  mood: CuprumMood;
  energy: CuprumEnergy;
  confidence: CuprumConfidence;
  dataState: CuprumDataState;
  riskCount: number;
  opportunityCount: number;
  briefing: CuprumBriefing;
}

export interface CuprumVoiceRuntime {
  bond: number;
  turn: number;
  freshData: boolean;
}

export type CuprumIntent =
  | 'spark'
  | 'sense'
  | 'sense-more'
  | 'sense-worry'
  | 'weight'
  | 'weight-change'
  | 'change'
  | 'driver-detail'
  | 'drivers'
  | 'driver-outlook'
  | 'memory-why'
  | 'risk'
  | 'opportunity'
  | 'board-read'
  | 'wires';

const MONTHS = [
  'January',
  'February',
  'March',
  'April',
  'May',
  'June',
  'July',
  'August',
  'September',
  'October',
  'November',
  'December',
];

const SPARK_BASE: Record<CuprumMood, string> = {
  molten_bull:
    "Loud. The whole board is warm, and I've been practicing my roar in the reflection of your coffee.",
  patina_bear:
    "Low and green. I'm keeping my edges soft until the weather makes up its mind.",
  tempered_neutral:
    "Steady. I've been listening to the space between prices. It has a nice hum.",
};

const DIRECTION_WORDS: Record<Exclude<CuprumDirection, 'unknown'>, string> = {
  warming: 'leaning in',
  cooling: 'easing back',
  steady: 'holding still',
};

const DIRECTION_SENTENCES: Record<CuprumDirection, string> = {
  warming: 'It is leaning in',
  cooling: 'It is easing back',
  steady: 'It is holding still',
  unknown: 'It is humming',
};

/** Below this percent move a signal counts as steady rather than leaning. */
const DIRECTION_EPSILON_PERCENT = 0.1;

/**
 * One bubble fits the companion's fixed stage. Longer lines open a scrollbar,
 * so every composed reply is trimmed at a sentence boundary before it ships.
 */
export const CUPRUM_SPEECH_BUDGET = 220;

/** Attention shares said the way a pet would say them: never a raw percent. */
const SHARE_WORDS = [
  'less than one in ten',
  'one in ten',
  'two in ten',
  'three in ten',
  'four in ten',
  'five in ten',
  'six in ten',
  'seven in ten',
  'eight in ten',
  'nine in ten',
  'all of it',
];

function shareWords(weight: number | null | undefined): string {
  if (typeof weight !== 'number' || !Number.isFinite(weight) || weight <= 0) {
    return 'a whisper';
  }
  if (weight < 0.05) return SHARE_WORDS[0];
  const tenths = Math.min(10, Math.max(1, Math.round(weight * 10)));
  return SHARE_WORDS[tenths];
}

function loudnessFromWeight(weight: number | null | undefined): CuprumDriverProfile['loudness'] {
  if (typeof weight !== 'number' || !Number.isFinite(weight)) return 'quiet';
  if (weight >= 0.15) return 'loud';
  if (weight >= 0.08) return 'middling';
  return 'quiet';
}

export function cleanFeatureLabel(
  label: string | null | undefined,
  feature: string | null | undefined,
): string {
  let raw = (label || feature || '').trim();
  if (!raw) return 'Copper';

  // Strip any redundant 1-day lag phrases already wrapped in parentheses
  raw = raw.replace(/\s*\((?:1-day\s+lag|return\s+lag\s*1(?:\s*days?)?)\)/gi, '').trim();

  // If label is purely a bare price change or 1-day lag phrase without an asset (e.g. "1-day price change", "return lag 1 day")
  if (
    /^(?:(\d+-day|daily)\s+(?:price\s+)?(?:change|return)|1\s*day\s*price|(?:1-day\s+)?lag(?:ged)?(?:\s+return)?(?:\s*1)?(?:\s*days?)?|return\s+lag\s*1(?:\s*days?)?|lag\s*1(?:\s*days?)?)/i.test(
      raw,
    )
  ) {
    return 'Copper';
  }

  // Handle "Subject · Indicator" or "Subject — Indicator"
  const split = raw.split(/\s+[·—–-]\s+/);
  if (split.length >= 2) {
    const rawSubject = split[0].trim();
    const indicator = split.slice(1).join(' ').trim();
    const subject = rawSubject.replace(/\s+Futures$/i, '');

    // 1-day return, 1-day price change, or 1-day lag: the subject alone is the natural human driver name (e.g. US Dollar Index, Copper, Vale, China Large-Cap)
    // 1-day lag / "return lag 1 day" is already that day's price/return, so omit redundant lag phrasing for Cuprum
    if (
      /^(?:(?:1-day|daily)\s+(?:price\s+)?(?:change|return)|(?:1-day\s+)?lag(?:ged)?(?:\s+return)?(?:\s*1)?(?:\s*days?)?|return\s+lag\s*1(?:\s*days?)?|lag\s*1(?:\s*days?)?)$/i.test(
        indicator,
      )
    ) {
      return subject;
    }

    // Meaningful technical metrics (averages, volatility, multi-day lag, momentum): keep concise in parentheses
    // e.g. "20-day moving average" -> "20-day average", "3-day lagged return" -> "3-day lag"
    const conciseIndicator = indicator
      .replace(/weighted\s+(?:moving\s+)?average/i, 'EMA')
      .replace(/moving\s+average/i, 'average')
      .replace(/price\s+volatility/i, 'volatility')
      .replace(/price\s+momentum/i, 'momentum')
      .replace(/return\s+lag\s*(\d+)\s*days?/i, '$1-day lag')
      .replace(/(\d+)-day\s+lagged\s+return/i, '$1-day lag')
      .replace(/lagged\s+return/i, 'lag');

    return `${subject} (${conciseIndicator})`;
  }

  // Fallback heuristic for raw ticker symbols:
  if (/^DX_?Y_?NYB/i.test(raw)) return 'US Dollar Index';
  if (/^HG[=_]F?_?(?:ret1|lag_ret1_1)?$/i.test(raw) || raw === 'ret1' || raw === 'lag_ret1_1') return 'Copper';
  if (/^GXC(?:_lag_ret1_1)?$/i.test(raw)) return 'China Large-Cap';
  if (/^FCX/i.test(raw)) {
    if (/vol/i.test(raw)) return 'Freeport-McMoRan (10-day volatility)';
    return 'Freeport-McMoRan';
  }
  if (/^SCCO/i.test(raw)) {
    if (/EMA/i.test(raw)) return 'Southern Copper (20-day average)';
    return 'Southern Copper';
  }

  return raw.replace(/_/g, ' ');
}

function buildDriverProfiles(
  signals: ExplainFeatureEntry[],
  fallbackHorizon: string | null,
): CuprumDriverProfile[] {
  return signals.slice(0, 3).map((signal) => {
    const rawWeight =
      typeof signal.weight === 'number' && Number.isFinite(signal.weight) ? signal.weight : 0;
    const weightPct = Math.round(rawWeight * 1000) / 10;
    return {
      phrase: signalPhrase(signal.label, signal.category, signal.feature),
      label: cleanFeatureLabel(signal.label, signal.feature),
      feature: signal.feature ?? '',
      weightPct,
      loudness: loudnessFromWeight(signal.weight),
      share: shareWords(signal.weight),
      direction: directionFromChange(signal.recent),
      horizon: signal.time_horizon ?? fallbackHorizon,
    };
  });
}

export function buildCuprumBriefing(
  commentary: CommentaryResponse | null,
  explain: ExplainResponse | null,
  tftAnalysis: TFTAnalysisResponse | null,
): CuprumBriefing {
  const companion = commentary?.companion_state ?? null;
  const highlights = companion?.xai_highlights ?? null;
  const signals: ExplainFeatureEntry[] =
    highlights?.top_signals ?? explain?.variable_selection?.encoder ?? [];
  const topSignal: ExplainFeatureEntry | null = signals[0] ?? null;
  const secondarySignal: ExplainFeatureEntry | null = signals[1] ?? null;
  const sections = commentary?.sections ?? null;
  const anchors = highlights?.memory_anchors ?? explain?.attention?.top_days ?? [];
  const memories = [
    ...new Set(
      anchors
        .map((anchor) => monthFromDate(anchor.date))
        .filter((month): month is string => Boolean(month)),
    ),
  ];
  const month = memories[0] ?? null;
  const staleness = tftAnalysis?.prediction?.baseline_staleness_days;
  const risks = uniqueClauses((sections?.risks ?? []).map(riskClause)).slice(0, 2);
  const opportunities = uniqueClauses(
    (sections?.opportunities ?? []).map(opportunityClause),
  ).slice(0, 2);
  const xaiBrief = sections?.xai_brief?.trim() ?? '';
  const driverPhrase = topSignal
    ? signalPhrase(topSignal.label, topSignal.category, topSignal.feature)
    : xaiBrief
      ? signalPhrase(xaiBrief, null)
      : null;
  const horizon =
    highlights?.horizon_label ??
    explain?.horizon?.label ??
    topSignal?.time_horizon ??
    tftAnalysis?.primary_horizon ??
    null;
  const narrative = sections?.synthesis?.trim() || explain?.notes?.trim() || '';

  return {
    driver: driverPhrase
      ? {
          phrase: driverPhrase,
          direction: directionFromChange(topSignal?.recent),
          trait: safeTrait(topSignal?.description ?? xaiBrief),
          horizon:
            topSignal?.time_horizon ?? highlights?.horizon_label ?? explain?.horizon?.label ?? null,
        }
      : null,
    secondary: secondarySignal
      ? {
          phrase: signalPhrase(
            secondarySignal.label,
            secondarySignal.category,
            secondarySignal.feature,
          ),
          direction: directionFromChange(secondarySignal.recent),
          trait: safeTrait(secondarySignal.description),
          horizon: secondarySignal.time_horizon ?? horizon,
        }
      : null,
    drivers: buildDriverProfiles(signals, horizon),
    memory: month ? { month, phrase: `a ${month} session` } : null,
    memories,
    risks,
    opportunities,
    synthesis: narrative ? synthesisClause(narrative) : null,
    biasWarning: Boolean(sections?.bias_warning),
    freshness:
      typeof staleness === 'number' ? (staleness <= 2 ? 'fresh' : 'stale') : 'unknown',
    anomaly: Boolean(tftAnalysis?.prediction?.anomaly_detected),
    horizon,
    horizonPhrase: horizonPhrase(horizon),
  };
}

export function composeCuprumReply(
  intent: CuprumIntent,
  ctx: CuprumVoiceContext,
  runtime: CuprumVoiceRuntime,
): string {
  if (ctx.dataState !== 'ready') return dataStateReply(intent, ctx.dataState);

  const briefing = ctx.briefing;
  const phrase = briefing.driver?.phrase ?? 'the hum';

  switch (intent) {
    case 'spark': {
      const base = SPARK_BASE[ctx.mood];
      if (!briefing.driver) return base;
      const direction =
        DIRECTION_WORDS[briefing.driver.direction as Exclude<CuprumDirection, 'unknown'>] ??
        'humming';
      return `${base} ${capitalize(phrase)} is ${direction}.`;
    }
    case 'sense': {
      const opening =
        ctx.mood === 'molten_bull'
          ? `There's a current in ${phrase}. It's warm. I'd bite it if I had more teeth.`
          : ctx.mood === 'patina_bear'
            ? `${capitalize(phrase)} is breathing slow. That's either patience or a trap.`
            : `I keep hearing ${phrase}. Not loud. Just... persistent.`;
      const balance = balanceSentence(ctx, briefing);
      return balance ? `${opening} ${balance}` : opening;
    }
    case 'sense-more': {
      const direction = briefing.driver?.direction ?? 'unknown';
      const parts = [
        `I keep looking at ${phrase}. ${DIRECTION_SENTENCES[direction]}.`,
      ];
      if (briefing.driver) parts.push(`${capitalize(briefing.driver.trait)}.`);
      if (briefing.memory) parts.push(`It rhymes with ${briefing.memory.phrase}.`);
      if (briefing.horizonPhrase) {
        parts.push(`Everything I hear is about ${briefing.horizonPhrase}.`);
      }
      if (briefing.biasWarning) parts.push("There's a footnote in the air.");
      return joinWithin(parts);
    }
    case 'sense-worry': {
      const knot = briefing.risks[0] ? ` ${capitalize(briefing.risks[0])}.` : '';
      if (ctx.confidence === 'high') {
        let speech = `I'd say it out loud: the metal is leaning, not promising. Leaning is not a promise.${knot}`;
        if (briefing.anomaly) speech += ' One of the gauges is blinking out of rhythm.';
        if (briefing.biasWarning) speech += " And there's a footnote in the air.";
        return speech;
      }
      if (ctx.confidence === 'low') {
        const stale =
          briefing.freshness === 'stale'
            ? ' The wires are a little stale, so I am holding this loosely.'
            : '';
        return `I'd whisper it. The wires are still deciding, and I don't like to interrupt.${stale}`;
      }
      return `I'd say it with one eye open. The metal is leaning. It hasn't picked a direction to fall.${knot}`;
    }
    case 'weight': {
      if (briefing.driver?.phrase === 'the paper weather') {
        return "The dollar keeps tapping on the window. I'm trying not to answer.";
      }
      if (ctx.riskCount > ctx.opportunityCount) {
        return `Something in ${phrase} is leaning on the door. I can hear the hinges.`;
      }
      if (ctx.opportunityCount > ctx.riskCount) {
        const door = briefing.opportunities[0] ?? 'a small door is still open';
        return `The doors are winning. ${capitalize(door)}. I'm keeping one eye on the hinges.`;
      }
      if (ctx.riskCount > 0 || ctx.opportunityCount > 0) {
        return 'A balanced knot. Those are the ones that bite.';
      }
      return `I don't have a knot to show you today. Just ${phrase}, humming.`;
    }
    case 'weight-change': {
      if (ctx.confidence === 'low') {
        return "Ask me again after the next whisper. I don't want to guess before the wires settle.";
      }
      return `A quieter ${phrase} would help. Or a louder one. I'm not picky. I just want it to stop pretending it isn't there.`;
    }
    case 'change': {
      if (runtime.freshData) {
        let speech = `The wind changed since you last looked. My attention slid to ${phrase}.`;
        if (briefing.anomaly) speech += ' One gauge is blinking out of rhythm.';
        if (briefing.freshness === 'stale') {
          speech += ' The wires are a little stale, so I am holding this loosely.';
        }
        return `${speech} I don't trust a mood that changes its shoes that quickly.`;
      }
      if (briefing.freshness === 'stale') {
        return "The wires are a little stale. Nothing has changed since you last looked, and I won't dress that up as a prophecy.";
      }
      return `Nothing new on the wires yet. I'm still holding ${phrase} in my teeth.`;
    }
    case 'driver-detail': {
      const parts = [
        `${capitalize(phrase)} is the loudest thing I hear, and I would rather be honest.`,
      ];
      if (briefing.driver) parts.push(`${capitalize(briefing.driver.trait)}.`);
      if (briefing.driver) parts.push(`${DIRECTION_SENTENCES[briefing.driver.direction]}.`);
      if (briefing.secondary) {
        parts.push(
          `Just behind it, ${briefing.secondary.phrase} is ${directionWord(
            briefing.secondary.direction,
          )} too.`,
        );
      }
      if (briefing.risks[0]) {
        parts.push(`The nearest knot: ${briefing.risks[0]}.`);
      }
      return joinWithin(parts);
    }
    case 'drivers': {
      const cast = briefing.drivers;
      if (cast.length === 0) {
        return "I don't hear anything loud enough to name yet. Just the hum, and me listening.";
      }
      const [lead, second, third] = cast;
      const parts = [
        `${capitalize(lead.phrase)} is the loudest — ${lead.share} of everything I hear, and it is ${directionWord(
          lead.direction,
        )}.`,
      ];
      if (second) {
        parts.push(
          `${capitalize(second.phrase)} sits right behind it, ${
            second.loudness === 'quiet' ? 'quieter but awake' : 'almost as loud'
          }.`,
        );
      }
      if (third) {
        parts.push(
          `Then ${third.phrase}, ${third.share}, ${directionWord(third.direction)}.`,
        );
      }
      return joinWithin(parts);
    }
    case 'driver-outlook': {
      const cast = briefing.drivers;
      if (cast.length === 0) {
        return "I only see this window, and there's nothing loud in it. Ask me again when the board speaks.";
      }
      const [lead, second] = cast;
      const parts = [
        `${capitalize(lead.phrase)} is still the loudest, still ${directionWord(lead.direction)}.`,
        'That is not a promise: it is only what the direction shows right now.',
      ];
      if (second) {
        parts.splice(1, 0, `And ${second.phrase} is ${directionWord(second.direction)} too.`);
      }
      return joinWithin(parts);
    }
    case 'memory-why': {
      if (!briefing.memory) {
        return "I don't have an old day to point at yet. The board is still making memories.";
      }
      let speech = `${capitalize(
        briefing.memory.phrase,
      )} keeps rhyming with today. I don't know if it's memory or a warning.`;
      const older = briefing.memories[1];
      if (older) {
        speech += ` So does ${article(older)} ${older} session.`;
      }
      return `${speech} The board remembers in patterns, not in numbers.`;
    }
    case 'risk': {
      const knot = briefing.risks[0] ?? 'a loose thread I cannot name yet';
      return `Here is the knot I keep touching: ${knot}. I don't call it a warning. I call it a knot with a good grip.`;
    }
    case 'opportunity': {
      const door = briefing.opportunities[0] ?? 'a small door is still open';
      return `Here is the door I keep seeing: ${door}. I'm not going to shove you through it.`;
    }
    case 'board-read': {
      const reading = briefing.synthesis ?? 'the board is humming, and it wants me to listen';
      return `The board reads like this: ${reading}. I never see the numbers it used to get there. I only get the shape of it.`;
    }
    case 'wires': {
      const cast = briefing.drivers;
      if (cast.length === 0) {
        return 'The wires are quiet right now. Nothing is pulling hard enough to register on the scales.';
      }
      const [lead, second, third] = cast;
      if (ctx.mood === 'molten_bull') {
        const parts = [
          `${lead.label} is eating ${lead.weightPct.toFixed(1)}% of the fire right now, ${directionWord(lead.direction)}.`,
        ];
        if (second) {
          parts.push(`Right behind it, ${second.label} claims ${second.weightPct.toFixed(1)}%.`);
        }
        if (third) {
          parts.push(
            `And ${third.label} takes ${third.weightPct.toFixed(1)}%. That's where the heat is coming from.`,
          );
        }
        return joinWithin(parts);
      }
      if (ctx.mood === 'patina_bear') {
        const parts = [
          `${lead.weightPct.toFixed(1)}% of my headache is ${lead.label}. It's ${directionWord(lead.direction)} like heavy damp air.`,
        ];
        if (second) {
          parts.push(`Another ${second.weightPct.toFixed(1)}% is ${second.label}.`);
        }
        if (third) {
          parts.push(`And ${third.weightPct.toFixed(1)}% is ${third.label}. It wears on the joints.`);
        }
        return joinWithin(parts);
      }
      const parts = [
        `The wires are humming with clear numbers: ${lead.label} holds ${lead.weightPct.toFixed(1)}% of the model's focus.`,
      ];
      if (second) {
        parts.push(`${second.label} follows at ${second.weightPct.toFixed(1)}%.`);
      }
      if (third) {
        parts.push(`And ${third.label} sits at ${third.weightPct.toFixed(1)}%. The scales are steady.`);
      }
      return joinWithin(parts);
    }
    default:
      return "I'm here. Listening.";
  }
}

export function signalPhrase(
  label: string | null | undefined,
  category: string | null | undefined,
  feature?: string | null,
): string {
  const haystack = `${label ?? ''} ${category ?? ''} ${feature ?? ''}`.toLowerCase();
  if (/dollar|dxy|usd|euro|currency|yen|yuan|fx/.test(haystack)) return 'the paper weather';
  if (/china|gxc|shfe|pmi|hang seng/.test(haystack)) return 'the far side of the world';
  if (/inventor|stock|warehouse|supply|premium|backwardation|contango/.test(haystack)) {
    return 'the warehouses';
  }
  if (/oil|energy|gas|power|electric|grid/.test(haystack)) return 'the hot wires';
  if (/yield|rate|fed|treasury|bond|inflation/.test(haystack)) return 'the tall rooms';
  if (/vol|vix|risk|uncertain/.test(haystack)) return 'the static';
  if (/momentum|trend|return|price|close|moving average/.test(haystack)) return 'the current';

  const byCategory: Record<string, string> = {
    Momentum: 'the current',
    Trend: 'the long tide',
    Volatility: 'the static',
    Sentiment: "the crowd's mood",
    Macro: 'the weather from far away',
    Sector: 'the neighboring rooms',
    Embedding: 'a pattern in the noise',
  };
  return byCategory[category ?? ''] ?? 'the hum';
}

function dataStateReply(intent: CuprumIntent, dataState: CuprumDataState): string {
  if (dataState === 'loading') {
    if (intent === 'spark') {
      return 'Still warming up. The wires are full of half-finished thoughts.';
    }
    if (intent === 'weight') return "The wires are quiet. I'm listening to the quiet.";
    if (intent === 'change') return 'Nothing new yet. The wires are still warm.';
    return "The wires are still warm. I can hear something moving, but it hasn't decided what it is yet.";
  }
  if (dataState === 'error') {
    if (intent === 'spark') return 'A little thin. Something in the wires went quiet.';
    if (intent === 'change') return "I lost the thread for a moment. I'm listening to the quiet.";
    return "The wires went quiet. I'm listening to the quiet.";
  }
  if (intent === 'spark') return "Quiet. But I'm a copper thing. We're good at waiting.";
  if (intent === 'change') return "The board is cold. Nothing has changed, and I won't pretend otherwise.";
  return "The board is cold. I'm listening anyway.";
}

function balanceSentence(ctx: CuprumVoiceContext, briefing: CuprumBriefing): string | null {
  if (ctx.riskCount === 0 && ctx.opportunityCount === 0) return null;
  if (ctx.riskCount > ctx.opportunityCount) {
    const knot = briefing.risks[0] ?? 'a loose thread I cannot name yet';
    return `More knots than doors today. ${capitalize(knot)}.`;
  }
  if (ctx.opportunityCount > ctx.riskCount) {
    const door = briefing.opportunities[0] ?? 'a small door is still open';
    return `The doors are winning. ${capitalize(door)}.`;
  }
  return 'A balanced knot. Those are the ones that bite.';
}

function directionFromChange(recent: ExplainRecent | null | undefined): CuprumDirection {
  const percent = relativeChangePercent(recent);
  if (percent === null) return 'unknown';
  if (percent > DIRECTION_EPSILON_PERCENT) return 'warming';
  if (percent < -DIRECTION_EPSILON_PERCENT) return 'cooling';
  return 'steady';
}

/**
 * The backend always reports `change` as an absolute delta and `change_pct` as
 * the percent move, so the direction is decided on a percentage. A signal
 * without a usable base cannot be turned into one and stays unknown.
 */
function relativeChangePercent(recent: ExplainRecent | null | undefined): number | null {
  if (!recent) return null;
  if (typeof recent.change_pct === 'number' && !Number.isNaN(recent.change_pct)) {
    return recent.change_pct;
  }
  if (typeof recent.change !== 'number' || Number.isNaN(recent.change)) return null;
  if (typeof recent.last_value !== 'number' || recent.last_value === 0) return null;
  return (recent.change / Math.abs(recent.last_value)) * 100;
}

function monthFromDate(date: string | null | undefined): string | null {
  const match = /^\d{4}-(\d{2})-\d{2}/.exec(date ?? '');
  if (!match) return null;
  return MONTHS[Number(match[1]) - 1] ?? null;
}

function riskClause(text: string): string {
  const haystack = text.toLowerCase();
  if (/dollar|dxy|usd|euro|currency|yen|yuan|fx/.test(haystack)) {
    return 'the paper weather is pulling at the seams';
  }
  if (/spread|backwardation|contango/.test(haystack)) {
    return 'the tight spreads are pulling both ways';
  }
  if (/china|gxc|shfe|pmi|hang seng|equit/.test(haystack)) {
    return 'the far side of the world is restless';
  }
  if (/smelter|restock|inventor|warehouse|stock|supply|prompt/.test(haystack)) {
    return 'the warehouses are leaning';
  }
  if (/oil|energy|gas|power|grid/.test(haystack)) return 'the hot wires are buzzing';
  if (/yield|rate|fed|treasury|bond|inflation/.test(haystack)) {
    return 'the tall rooms are making noise';
  }
  if (/vol|vix|risk|uncertain/.test(haystack)) return 'the static is rising';
  if (/momentum|trend|return|price|close|moving average/.test(haystack)) {
    return 'the current is uneven';
  }
  return 'a loose thread I cannot name yet';
}

function opportunityClause(text: string): string {
  const haystack = text.toLowerCase();
  if (/spread|backwardation|contango/.test(haystack)) {
    return 'the tight spreads are holding a door open';
  }
  if (/dollar|dxy|usd|euro|currency|yen|yuan|fx/.test(haystack)) {
    return 'the paper weather is easing';
  }
  if (/china|gxc|shfe|pmi|hang seng|equit/.test(haystack)) {
    return 'the far side of the world has a little push';
  }
  if (/inventor|warehouse|stock|supply|draw/.test(haystack)) {
    return 'the warehouses are opening a door';
  }
  if (/demand|consumption|buyer/.test(haystack)) {
    return 'someone is knocking on the demand door';
  }
  if (/momentum|trend|return|price|close|moving average/.test(haystack)) {
    return 'the current has a little push behind it';
  }
  return 'a small door is still open';
}

function safeTrait(description: string | null | undefined): string {
  const haystack = (description ?? '').toLowerCase();
  if (/cap.*upside|stronger dollar|capp/.test(haystack)) {
    return 'it has a history of pulling the roof down';
  }
  if (/spread|tight/.test(haystack)) return 'it hums like a held breath';
  if (/demand|equity|momentum/.test(haystack)) {
    return 'it carries a little push from somewhere else';
  }
  if (/restock|inventory|supply|warehouse/.test(haystack)) {
    return 'it keeps moving the furniture';
  }
  if (/volatil|risk|uncertain/.test(haystack)) return 'it does not sit still';
  return 'it has a long memory';
}

/** The console narrative, told without its numbers. */
function synthesisClause(text: string): string {
  const haystack = text.toLowerCase();
  if (/spread|backwardation|contango/.test(haystack)) {
    return 'the spreads are holding their breath';
  }
  if (/dollar|dxy|usd|euro|currency|yen|yuan|fx/.test(haystack)) {
    return 'the paper weather is doing most of the talking';
  }
  if (/china|gxc|shfe|pmi|hang seng|equit/.test(haystack)) {
    return 'the far side of the world is moving';
  }
  if (/restock|inventor|warehouse|stock|supply/.test(haystack)) {
    return 'the warehouses are shifting quietly';
  }
  if (/demand|consumption|buyer/.test(haystack)) {
    return 'someone is still knocking on the demand door';
  }
  if (/momentum|trend|constructive|support|posture/.test(haystack)) {
    return 'the current is still running';
  }
  if (/volatil|risk|uncertain|vol/.test(haystack)) return 'the static is up a little';
  return 'the board is humming, and it wants me to listen';
}

/** The forecast window, in words the reader can picture. */
function horizonPhrase(label: string | null): string | null {
  const haystack = (label ?? '').toLowerCase();
  if (!haystack) return null;
  if (/week/.test(haystack)) return 'the coming week';
  if (/month/.test(haystack)) return 'the coming month';
  if (/quarter/.test(haystack)) return 'the coming quarter';
  if (/three|3d/.test(haystack)) return 'the next three sessions';
  if (/five|5d/.test(haystack)) return 'the next five sessions';
  if (/day|1d|daily|tomorrow/.test(haystack)) return 'the next session';
  return null;
}

function article(month: string): string {
  return /^[aeiou]/i.test(month) ? 'an' : 'a';
}

function uniqueClauses(clauses: string[]): string[] {
  return [...new Set(clauses)];
}

function capitalize(value: string): string {
  return value.length > 0 ? `${value[0].toUpperCase()}${value.slice(1)}` : value;
}

function directionWord(direction: CuprumDirection): string {
  return direction === 'unknown' ? 'humming' : DIRECTION_WORDS[direction];
}

/**
 * Keeps the bubble inside its stage: parts are added in priority order and the
 * line stops before the first part that would overflow. Nothing is cut midway.
 */
function joinWithin(
  parts: string[],
  budget: number = CUPRUM_SPEECH_BUDGET,
): string {
  let speech = '';
  for (const part of parts) {
    const candidate = speech ? `${speech} ${part}` : part;
    if (candidate.length > budget) {
      if (!speech) {
        return candidate.length > budget
          ? candidate.slice(0, budget - 1).replace(/\s+\S*$/, '\u2026')
          : candidate;
      }
      break;
    }
    speech = candidate;
  }
  return speech;
}
