import type {
  CommentaryResponse,
  ExplainFeatureEntry,
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

export interface CuprumBriefing {
  driver: CuprumDriverBriefing | null;
  memory: CuprumMemoryBriefing | null;
  risks: string[];
  opportunities: string[];
  biasWarning: boolean;
  freshness: CuprumFreshness;
  anomaly: boolean;
  horizon: string | null;
}

/** The slice of dialogue state the local voice needs. */
export interface CuprumVoiceContext {
  mood: CuprumMood;
  energy: CuprumEnergy;
  confidence: CuprumConfidence;
  dataState: CuprumDataState;
  riskCount: number;
  opportunityCount: number;
  hasBiasWarning: boolean;
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
  | 'memory-why'
  | 'risk'
  | 'opportunity';

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

export function buildCuprumBriefing(
  commentary: CommentaryResponse | null,
  explain: ExplainResponse | null,
  tftAnalysis: TFTAnalysisResponse | null,
): CuprumBriefing {
  const companion = commentary?.companion_state ?? null;
  const highlights = companion?.xai_highlights ?? null;
  const topSignal: ExplainFeatureEntry | null =
    highlights?.top_signals?.[0] ??
    explain?.variable_selection?.encoder?.[0] ??
    null;
  const sections = commentary?.sections ?? null;
  const anchorDate =
    highlights?.memory_anchors?.[0]?.date ??
    explain?.attention?.top_days?.[0]?.date ??
    null;
  const month = monthFromDate(anchorDate);
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

  return {
    driver: driverPhrase
      ? {
          phrase: driverPhrase,
          direction: directionFromChange(topSignal?.recent?.change),
          trait: safeTrait(topSignal?.description ?? xaiBrief),
          horizon:
            topSignal?.time_horizon ??
            highlights?.horizon_label ??
            explain?.horizon?.label ??
            tftAnalysis?.primary_horizon ??
            null,
        }
      : null,
    memory: month ? { month, phrase: `a ${month} session` } : null,
    risks,
    opportunities,
    biasWarning: Boolean(sections?.bias_warning),
    freshness:
      typeof staleness === 'number' ? (staleness <= 2 ? 'fresh' : 'stale') : 'unknown',
    anomaly: Boolean(tftAnalysis?.prediction?.anomaly_detected),
    horizon:
      highlights?.horizon_label ??
      explain?.horizon?.label ??
      topSignal?.time_horizon ??
      tftAnalysis?.primary_horizon ??
      null,
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
      let speech = `I keep looking at ${phrase}. ${DIRECTION_SENTENCES[briefing.driver?.direction ?? 'unknown']}.`;
      if (briefing.driver) speech += ` ${capitalize(briefing.driver.trait)}.`;
      if (briefing.memory) speech += ` It rhymes with ${briefing.memory.phrase}.`;
      if (briefing.biasWarning) {
        speech += " There's a footnote in the air. I don't like footnotes.";
      }
      return speech;
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
      let speech = `${capitalize(phrase)} is the loudest thing I hear. ${capitalize(
        briefing.driver?.trait ?? 'it has a long memory',
      )}.`;
      if (briefing.driver) {
        speech += ` ${DIRECTION_SENTENCES[briefing.driver.direction]}.`;
      }
      if (briefing.risks[0]) {
        speech += ` The nearest knot: ${briefing.risks[0]}.`;
      }
      return `${speech} I don't need it to be right; I need it to be honest.`;
    }
    case 'memory-why': {
      if (!briefing.memory) {
        return "I don't have an old day to point at yet. The board is still making memories.";
      }
      return `${capitalize(
        briefing.memory.phrase,
      )} keeps rhyming with today. I don't know if it's memory or a warning. The board remembers in patterns, not in numbers.`;
    }
    case 'risk': {
      const knot = briefing.risks[0] ?? 'a loose thread I cannot name yet';
      return `Here is the knot I keep touching: ${knot}. I don't call it a warning. I call it a knot with a good grip.`;
    }
    case 'opportunity': {
      const door = briefing.opportunities[0] ?? 'a small door is still open';
      return `Here is the door I keep seeing: ${door}. I'm not going to shove you through it.`;
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

function directionFromChange(change: number | null | undefined): CuprumDirection {
  if (typeof change !== 'number' || Number.isNaN(change)) return 'unknown';
  if (change > 0.0001) return 'warming';
  if (change < -0.0001) return 'cooling';
  return 'steady';
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

function uniqueClauses(clauses: string[]): string[] {
  return [...new Set(clauses)];
}

function capitalize(value: string): string {
  return value.length > 0 ? `${value[0].toUpperCase()}${value.slice(1)}` : value;
}
