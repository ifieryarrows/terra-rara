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

export interface CuprumBriefing {
  driver: CuprumDriverBriefing | null;
  /** The next loudest encoder signal, when the console sent more than one. */
  secondary: CuprumDriverBriefing | null;
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
  | 'memory-why'
  | 'risk'
  | 'opportunity'
  | 'board-read';

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
      let speech = `I keep looking at ${phrase}. ${DIRECTION_SENTENCES[briefing.driver?.direction ?? 'unknown']}.`;
      if (briefing.driver) speech += ` ${capitalize(briefing.driver.trait)}.`;
      if (briefing.memory) speech += ` It rhymes with ${briefing.memory.phrase}.`;
      if (briefing.horizonPhrase) {
        speech += ` Everything I am listening to is about ${briefing.horizonPhrase}.`;
      }
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
      if (briefing.secondary) {
        const direction =
          DIRECTION_WORDS[
            briefing.secondary.direction as Exclude<CuprumDirection, 'unknown'>
          ] ?? 'humming';
        speech += ` Behind it, ${briefing.secondary.phrase} is ${direction} too.`;
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
