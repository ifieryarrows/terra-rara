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
  /** A plain-language name for the input the model focused on. */
  label: string;
  /** Safe imagery used in speech, e.g. "the paper weather". */
  phrase: string;
  /** Recent movement of the input feature, not the forecast direction. */
  direction: CuprumDirection;
  horizon: string | null;
}

export interface CuprumMemoryBriefing {
  /** Month name only. The exact date never reaches the speech layer. */
  month: string;
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
  /** API context completeness, not confidence in the forecast. */
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
    "Bright. I've got an ambitious idea for this board, and I want to see if it can keep up.",
  patina_bear:
    "Thin. The board has already made me sigh twice. Let's keep this short and honest.",
  tempered_neutral:
    "Steady. I'm listening for the small clue under the loud ones.",
};

const RECENT_MOVEMENT: Record<CuprumDirection, string> = {
  warming: 'has moved up lately',
  cooling: 'has moved down lately',
  steady: 'has barely moved lately',
  unknown: 'has no clear recent move I can name',
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
          label: topSignal
            ? accessibleSignalName(topSignal)
            : 'the focus note in this snapshot',
          phrase: driverPhrase,
          direction: directionFromChange(topSignal?.recent),
          horizon:
            topSignal?.time_horizon ?? highlights?.horizon_label ?? explain?.horizon?.label ?? null,
        }
      : null,
    secondary: secondarySignal
      ? {
          label: accessibleSignalName(secondarySignal),
          phrase: signalPhrase(
            secondarySignal.label,
            secondarySignal.category,
            secondarySignal.feature,
          ),
          direction: directionFromChange(secondarySignal.recent),
          horizon: secondarySignal.time_horizon ?? horizon,
        }
      : null,
    memory: month ? { month } : null,
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
  if (ctx.dataState !== 'ready') return dataStateReply(intent, ctx.dataState, ctx.mood);

  const briefing = ctx.briefing;
  const phrase = briefing.driver?.phrase ?? 'the hum';

  switch (intent) {
    case 'spark': {
      const base = SPARK_BASE[ctx.mood];
      if (!briefing.driver) return base;
      return `${base} The clearest signal in this snapshot is ${briefing.driver.label}; it ${RECENT_MOVEMENT[briefing.driver.direction]}.`;
    }
    case 'sense': {
      const opening =
        ctx.mood === 'molten_bull'
          ? `There's a current in ${phrase}. I want to chase it, after I check where it bends.`
          : ctx.mood === 'patina_bear'
            ? `${capitalize(phrase)} is dragging its feet. So am I. Let's keep the question small.`
            : `I keep hearing ${phrase}. Not loud. Just... persistent.`;
      const balance = balanceSentence(ctx, briefing);
      return balance ? `${opening} ${balance}` : opening;
    }
    case 'sense-more': {
      let speech = briefing.driver
        ? `The strongest signal in this explanation is ${briefing.driver.label}. I call it ${briefing.driver.phrase}.`
        : 'I do not have a clear signal to show you in this snapshot.';
      if (briefing.driver) {
        speech += ` That input ${RECENT_MOVEMENT[briefing.driver.direction]}.`;
      }
      if (briefing.secondary) {
        speech += ` A second signal in the explanation is ${briefing.secondary.label}.`;
      }
      if (briefing.memory) {
        speech += ` It also looked closely at older data from ${briefing.memory.month}. That is where its attention went, not a promise that the same story will repeat.`;
      }
      if (briefing.horizonPhrase) {
        speech += ` The outlook covers ${briefing.horizonPhrase}.`;
      }
      if (briefing.biasWarning) {
        speech += ' There is also a bias note attached, so this view deserves an extra careful read.';
      }
      speech += ` ${moodFocusCaveat(ctx.mood)}`;
      return speech;
    }
    case 'sense-worry': {
      const knot = briefing.risks[0] ? ` ${capitalize(briefing.risks[0])}.` : '';
      if (ctx.confidence === 'high') {
        let speech = `I can say this out loud because I have commentary and a model explanation to work from; that still does not make the forecast certain. The metal is leaning, not promising.${knot}`;
        if (briefing.anomaly) speech += ' One of the gauges is blinking out of rhythm.';
        if (briefing.biasWarning) speech += ' There is a bias note attached, too.';
        return speech;
      }
      if (ctx.confidence === 'low') {
        const stale =
          briefing.freshness === 'stale'
            ? ' The data behind this view is also a little old.'
            : '';
        return `I'd whisper it: I only have a thin snapshot, so I won't guess at what comes next.${stale}`;
      }
      return `I'd say it with one eye open: I have some detail, but not a complete picture. The metal is leaning, not promising.${knot}`;
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
        return "This snapshot is thin. I'd need a fuller one before I could name what would change my view.";
      }
      if (ctx.mood === 'molten_bull') {
        return `A fresh clue that pushes against ${phrase} would make me slow down. I like a bold idea, but it still has to earn its place.`;
      }
      if (ctx.mood === 'patina_bear') {
        return `A clear update in ${phrase} might wake me up. Until then, I'm not chasing every flicker.`;
      }
      return `A clear new reading on ${phrase} would make me listen again. One update is a reason to look, not a reason to be certain.`;
    }
    case 'change': {
      if (runtime.freshData) {
        let speech = briefing.driver
          ? `A new snapshot arrived since you last looked. The wind changed in my little room; my clearest signal is ${briefing.driver.label}. That update alone does not mean the market moved.`
          : "A new snapshot arrived since you last looked. The wind changed in my little room, but I don't have a clear signal to point to yet.";
        if (briefing.anomaly) speech += ' One gauge is blinking out of rhythm.';
        if (briefing.freshness === 'stale') {
          speech += ' The data behind this view is a little old, so I am holding it loosely.';
        }
        const moodLine = ctx.mood === 'molten_bull'
          ? 'A new clue gives me something to chase, not a reason to declare victory.'
          : ctx.mood === 'patina_bear'
            ? "I'm too tired to chase every update; I'll wait for a clearer one."
            : 'A new snapshot is a new reading, not a guarantee.';
        return `${speech} ${moodLine}`;
      }
      if (briefing.freshness === 'stale') {
        return "The data behind this view is a little old. I can't tell you what has changed since you last looked, and I won't dress that up as a prophecy.";
      }
      return `No newer snapshot has arrived yet. I'm still holding ${briefing.driver?.label ?? phrase} in my teeth.`;
    }
    case 'driver-detail': {
      if (!briefing.driver) {
        return "I don't have a clear model clue in this snapshot, so I won't invent one.";
      }
      const focus = briefing.driver.label;
      let speech = `The strongest input signal in this snapshot's explanation is ${focus}.`;
      if (briefing.driver) {
        speech += ` I call it ${briefing.driver.phrase}.`;
        speech += ` That input ${RECENT_MOVEMENT[briefing.driver.direction]}.`;
      }
      if (briefing.secondary) {
        speech += ` A second signal in the explanation is ${briefing.secondary.label}.`;
      }
      if (briefing.risks[0]) {
        speech += ` One risk note in the same briefing: ${briefing.risks[0]}.`;
      }
      if (briefing.horizonPhrase) speech += ` This outlook covers ${briefing.horizonPhrase}.`;
      return `${speech} ${moodFocusCaveat(ctx.mood)}`;
    }
    case 'memory-why': {
      if (!briefing.memory) {
        return "I don't have a clear past date to point to in this snapshot. I won't invent a memory for the board.";
      }
      let speech = `The model's attention was strongest around ${briefing.memory.month} in the earlier data. That tells us where it looked, not that the same story will happen again.`;
      const older = briefing.memories[1];
      if (older) {
        speech += ` It also looked toward ${older}.`;
      }
      if (briefing.horizonPhrase) speech += ` It used those clues for an outlook covering ${briefing.horizonPhrase}.`;
      return `${speech} ${memoryMoodLine(ctx.mood)}`;
    }
    case 'risk': {
      if (!briefing.risks[0]) {
        return "I don't have a specific risk note in this snapshot. No point making up a knot.";
      }
      const knot = briefing.risks[0];
      const voice = ctx.mood === 'molten_bull'
        ? 'I would rather spot it early than let my enthusiasm hide it.'
        : ctx.mood === 'patina_bear'
          ? 'That is the part making my edges feel heavier.'
          : 'It is something to keep in view, not a promise that it will happen.';
      return `${capitalize(knot)} That's one risk the snapshot flags. ${voice}`;
    }
    case 'opportunity': {
      if (!briefing.opportunities[0]) {
        return "I don't have a specific hopeful note in this snapshot. I won't paint an opening onto a closed door.";
      }
      const door = briefing.opportunities[0];
      const voice = ctx.mood === 'molten_bull'
        ? 'I like the opening. I still want to know what could close it.'
        : ctx.mood === 'patina_bear'
          ? 'A possible bright spot. I will try not to sigh at it.'
          : 'A possible opening, not a guarantee.';
      return `${capitalize(door)} That's one possible bright spot in the snapshot. ${voice}`;
    }
    case 'board-read': {
      if (!briefing.synthesis) {
        return "I don't have a summary for this snapshot yet. I can still show you the feature signal I have.";
      }
      const reading = briefing.synthesis;
      return `The board reads like this: ${reading}. The model's explanation also shows which clues stood out; it does not prove what caused a future move.`;
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

function accessibleSignalName(signal: ExplainFeatureEntry): string {
  const source = `${signal.label ?? ''} ${signal.category ?? ''} ${signal.feature ?? ''}`.toLowerCase();
  if (/dollar|dxy|usd|euro|currency|yen|yuan|\bfx\b/.test(source)) return "the dollar's strength";
  if (/china|gxc|shfe|pmi|hang seng/.test(source)) return 'activity in China';
  if (/inventor|warehouse|stock|supply|draw|backwardation|contango/.test(source)) {
    return 'copper stored in warehouses';
  }
  if (/oil|energy|gas|power|electric|grid/.test(source)) return 'energy costs and supply';
  if (/yield|rate|fed|treasury|bond|inflation/.test(source)) return 'borrowing costs and inflation';
  if (/vol|vix|risk|uncertain/.test(source)) return 'how jumpy the market feels';
  if (/sentiment|emotion/.test(source)) return "the market's mood";
  if (/momentum|trend|return|price|close|moving average/.test(source)) {
    return 'recent copper price movement';
  }

  const categoryNames: Record<string, string> = {
    Macro: 'wider economic conditions',
    Momentum: 'recent copper price movement',
    Trend: 'the direction copper has been heading',
    Volatility: 'how jumpy the market feels',
    Sentiment: "the market's mood",
    Sector: 'activity in nearby industries',
    Embedding: 'a less familiar pattern in the data',
  };
  const byCategory = categoryNames[signal.category ?? ''];
  if (byCategory) return byCategory;

  const cleaned = (signal.label ?? signal.feature ?? '')
    .replace(/^[A-Z0-9=.:-]+[_:]/i, '')
    .replace(/[_/]+/g, ' ')
    .replace(/\b(?:sma|ema)\b/gi, 'moving average')
    .replace(/\bvolume\b/gi, 'trading activity')
    .replace(/\b(?:close|closing)\b/gi, 'copper price')
    .replace(/\b\d+(?:d|day)?\b/gi, '')
    .replace(/\s+/g, ' ')
    .trim();
  return cleaned || 'another market signal';
}

function moodFocusCaveat(mood: CuprumMood): string {
  if (mood === 'molten_bull') {
    return "I like a bold clue, but a model signal is not proof that it caused the forecast.";
  }
  if (mood === 'patina_bear') {
    return "That's enough to notice. A model signal is not proof of cause, however tired I am of the guesswork.";
  }
  return "A model signal shows what stood out in the explanation, not what caused the forecast.";
}

function memoryMoodLine(mood: CuprumMood): string {
  if (mood === 'molten_bull') return 'Patterns tempt me to run ahead; a past day is still no promise.';
  if (mood === 'patina_bear') return "I'm not calling it a rerun. Markets repeat just enough to be annoying.";
  return 'A historical focus is context, not a prediction that history will repeat.';
}

function dataStateReply(
  intent: CuprumIntent,
  dataState: CuprumDataState,
  mood: CuprumMood,
): string {
  if (dataState === 'loading') {
    if (intent === 'spark') {
      if (mood === 'molten_bull') return 'Still warming up. I have a plan ready; the wires are taking their time.';
      if (mood === 'patina_bear') return "Still warming up. Wake me when the snapshot is ready; the wires can wait.";
      return 'Still warming up. The wires are full of half-finished thoughts.';
    }
    if (intent === 'weight') return "The wires are quiet. I'm listening to the quiet.";
    if (intent === 'change') return 'Nothing new yet. The wires are still warm.';
    return "The wires are still warm. I can hear something moving, but it hasn't decided what it is yet.";
  }
  if (dataState === 'error') {
    if (intent === 'spark') {
      if (mood === 'molten_bull') return 'A little thin. The wires went quiet, but I have not lost my ambition.';
      if (mood === 'patina_bear') return 'A little thin. The wires went quiet, and honestly, so did I.';
      return 'A little thin. Something in the wires went quiet.';
    }
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
    return 'nearby copper supply may be getting tighter';
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
    return 'nearby copper supply looks tight';
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

/** The console narrative, told without its numbers. */
function synthesisClause(text: string): string {
  const haystack = text.toLowerCase();
  if (/spread|backwardation|contango/.test(haystack)) {
    return 'nearby copper supply looks tight';
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

function uniqueClauses(clauses: string[]): string[] {
  return [...new Set(clauses)];
}

function capitalize(value: string): string {
  return value.length > 0 ? `${value[0].toUpperCase()}${value.slice(1)}` : value;
}
