import type {
  CommentaryResponse,
  ExplainResponse,
  TFTAnalysisResponse,
} from '../types';

export type CuprumMood = 'molten_bull' | 'patina_bear' | 'tempered_neutral';
export type CuprumEnergy = 'calm' | 'charged' | 'volatile';
export type CuprumConfidence = 'high' | 'medium' | 'low';
export type CuprumDataState = 'ready' | 'loading' | 'empty' | 'error';

export interface CuprumContext {
  mood: CuprumMood;
  energy: CuprumEnergy;
  confidence: CuprumConfidence;
  stance: 'BULLISH' | 'BEARISH' | 'NEUTRAL';
  signalCategory: string | null;
  signalLabel: string | null;
  riskCount: number;
  opportunityCount: number;
  hasBiasWarning: boolean;
  dataState: CuprumDataState;
  updatedAt: string | null;
}

export interface CuprumRuntime {
  bond: number;
  storyIndex: number;
  turn: number;
  freshData: boolean;
}

export type CuprumChoiceEffect = 'warm' | 'tease' | 'curious';

export interface CuprumChoice {
  id: string;
  label: string;
  next?: string;
  close?: boolean;
  effect?: CuprumChoiceEffect;
  advanceStory?: boolean;
}

export interface CuprumNode {
  id: string;
  speech: string | ((ctx: CuprumContext, runtime: CuprumRuntime) => string);
  action?: string | ((ctx: CuprumContext, runtime: CuprumRuntime) => string | null);
  choices: CuprumChoice[];
}

export interface ResolvedCuprumNode {
  id: string;
  speech: string;
  action: string | null;
  choices: CuprumChoice[];
}

export interface CuprumStory {
  id: string;
  text: string;
  reflection: string;
}

export interface CuprumIdleLine {
  id: string;
  text: string;
}

export const ROOT_NODE_ID = 'root';

export const MAX_BOND = 9;

const GREETINGS: Record<CuprumMood, string> = {
  molten_bull: "You caught me mid-roar. Don't tell the bear.",
  patina_bear: "Mind the green. It's soft today, but it stains.",
  tempered_neutral: 'You came at a good moment. I was counting the space between prices.',
};

const BOND_GREETINGS: Record<CuprumMood, string> = {
  molten_bull: 'You came back. I kept the forge warm.',
  patina_bear: "You came back. The green is softer when you're here.",
  tempered_neutral: 'You came back. I saved you the quiet seat.',
};

const SPARK_LINES: Record<CuprumMood, string> = {
  molten_bull:
    "Loud. The whole board is warm, and I've been practicing my roar in the reflection of your coffee.",
  patina_bear:
    "Low and green. I'm keeping my edges soft until the weather makes up its mind.",
  tempered_neutral:
    "Steady. I've been listening to the space between prices. It has a nice hum.",
};

const SPARK_ACTIONS: Record<CuprumMood, string> = {
  molten_bull: 'The ember in its chest brightens.',
  patina_bear: 'A little green flame curls around its ear.',
  tempered_neutral: 'It taps one claw against the glass, very softly.',
};

const BALANCE_LINES = {
  knots: 'More knots than doors today.',
  doors: "The doors are winning. I'm watching the hinges.",
  balanced: 'A balanced knot. Those are the ones that bite.',
} as const;

export const CUPRUM_STORIES: CuprumStory[] = [
  {
    id: 'smiths',
    text: "The first smiths didn't trust copper. It kept changing color when the fire got too honest.",
    reflection:
      "Because it means I'm allowed to be different on different days. Most things aren't.",
  },
  {
    id: 'liberty',
    text: 'The Statue of Liberty is a copper girl wearing a hundred years of weather.',
    reflection:
      "I like her. She started bright and ended up green. That's not a failure. That's a life.",
  },
  {
    id: 'bell',
    text: 'A bell is a copper cousin with a job.',
    reflection:
      'All that metal, and it only gets to speak when someone else pulls the rope. I find that terrifying.',
  },
  {
    id: 'sea',
    text: 'The sea keeps a little copper in it. Not enough to make a coin, but enough to remember.',
    reflection: 'The sea is a very old pocket. I respect that.',
  },
  {
    id: 'first-metal',
    text: "Copper was one of the first metals people pulled out of stone. We've been in your pockets ever since.",
    reflection:
      "You carry us around and forget we're there. We're used to it. Mostly.",
  },
  {
    id: 'patina',
    text: "The green on my edges isn't damage. It's a diary.",
    reflection: "Every damp morning writes a line. I don't always like the entries.",
  },
  {
    id: 'lightning',
    text: "Lightning likes copper. It's the only gossip that travels faster than me.",
    reflection:
      "I've never been struck. I imagine it's like being interrupted by a very bright idea.",
  },
  {
    id: 'coin',
    text: 'A copper coin once bought a whole meal. Now it buys a rumor. I liked the meal better.',
    reflection:
      'Things get more valuable and less useful at the same time. I try not to take it personally.',
  },
  {
    id: 'antimicrobial',
    text: 'They say copper can kill little things on contact. I try not to think about it.',
    reflection: "I'd rather be a door handle than a weapon. Most of us would.",
  },
  {
    id: 'venus',
    text: 'The old alchemists gave copper the symbol of Venus. They thought beauty and heat belonged together.',
    reflection: 'They were right about the heat. The beauty is a matter of taste.',
  },
];

const IDLE_MOOD_LINES: Record<CuprumMood, CuprumIdleLine[]> = {
  molten_bull: [
    { id: 'bull-wrestle', text: 'I could take the whole board in a wrestling match. Politely.' },
    { id: 'bull-enthusiasm', text: 'Heat is just enthusiasm with a good publicist.' },
    { id: 'bull-teeth', text: 'I like it when the numbers run. It makes my teeth hum.' },
    {
      id: 'bull-coffee',
      text: "Don't tell anyone, but I practiced my roar in the reflection of your coffee.",
    },
    {
      id: 'bull-close',
      text: "There's a little fire in everything. Most people just don't stand close enough.",
    },
    { id: 'bull-tail', text: "If I had a tail, I'd wag it. I have a patina. It's similar." },
  ],
  patina_bear: [
    { id: 'bear-patinated', text: "I'm not gloomy. I'm patinated. There's a difference." },
    { id: 'bear-green', text: 'The green is where I keep the quiet days.' },
    { id: 'bear-rain', text: 'Everyone wants the bright metal. Nobody asks what the rain did to it.' },
    { id: 'bear-armor', text: 'Slow is a kind of armor. It gives you time to see the crack.' },
    { id: 'bear-cold', text: "I don't mind the cold. I mind the people who pretend it isn't cold." },
    {
      id: 'bear-sit',
      text: "You can sit here. I won't say anything. That's the nicest thing I do.",
    },
  ],
  tempered_neutral: [
    {
      id: 'neutral-hum',
      text: "I've been counting the space between prices. It has a nice hum.",
    },
    { id: 'neutral-weather', text: 'A market is just weather with paperwork.' },
    { id: 'neutral-notice', text: "I don't predict. I notice. It's a better job." },
    { id: 'neutral-middle', text: 'The middle is a good place to listen from.' },
    {
      id: 'neutral-wire',
      text: 'Everything is connected. Some things are just better at hiding the wire.',
    },
    { id: 'neutral-quiet', text: 'I like a quiet board. It gives my thoughts somewhere to sit.' },
  ],
};

const IDLE_GENERAL_LINES: CuprumIdleLine[] = [
  { id: 'general-bronze', text: 'Bronze is just copper that got ambitious.' },
  { id: 'general-screen', text: 'Your screen is warm. I like that.' },
  {
    id: 'general-storm',
    text: "There's a storm in the wires tonight. I can't tell if it's weather or people.",
  },
  { id: 'general-blink', text: "If you're tired, blink twice. I'll pretend not to notice." },
  {
    id: 'general-mornings',
    text: "I've seen a thousand mornings. The good ones start with someone saying 'just one more look.'",
  },
  {
    id: 'general-coin',
    text: 'A copper coin once bought a whole meal. Now it buys a rumor. I liked the meal better.',
  },
  {
    id: 'general-smiths',
    text: 'The old smiths used to say copper tells the truth about heat. People are harder.',
  },
  {
    id: 'general-wires',
    text: "I'm small, but I'm in every wire in this room. That counts for something.",
  },
  { id: 'general-sitting', text: "You don't have to say anything. I'm good at sitting." },
  { id: 'general-patina', text: "I like the word 'patina.' It sounds like a country I've never visited." },
  {
    id: 'general-heartbeat',
    text: "If you listen closely, the board has a heartbeat. It's mostly other people's.",
  },
  { id: 'general-promise', text: 'Every wire is a promise that something will arrive.' },
  {
    id: 'general-between',
    text: "The best things in a market happen between the lines. That's where I live.",
  },
];

const IDLE_BOND_LINES: CuprumIdleLine[] = [
  { id: 'bond-questions', text: 'You ask better questions than the board does.' },
  { id: 'bond-cloud', text: 'I saved you the good cloud. The one shaped like a coin.' },
  { id: 'bond-laptop', text: "I don't say this to everyone: you're allowed to close the laptop." },
  { id: 'bond-forge', text: 'You came back. I kept the forge warm.' },
  { id: 'bond-thought', text: "I'm still thinking about what you said. It's a good thought. It has weight." },
];

const IDLE_DATA_LINES: Record<Exclude<CuprumDataState, 'ready'>, CuprumIdleLine[]> = {
  loading: [
    { id: 'loading-warm', text: "The wires are still warm. I'm listening." },
    { id: 'loading-half', text: "Half-finished thoughts are the loudest. Give me a moment." },
  ],
  empty: [
    { id: 'empty-here', text: "The board is quiet. I'm still here." },
    { id: 'empty-cold', text: "The wires are cold. I'm listening anyway." },
  ],
  error: [
    { id: 'error-thread', text: 'I lost the thread for a moment. Sit with me anyway.' },
    { id: 'error-quiet', text: "Something in the wires went quiet. I don't like it either." },
  ],
};

function settleStoryLine(): CuprumChoice {
  return { id: 'story-installment', label: 'Tell me another.', next: 'story', effect: 'curious', advanceStory: true };
}

function askAgainChoice(): CuprumChoice {
  return { id: 'ask-again', label: 'Ask me something else.', next: ROOT_NODE_ID };
}

function tellStoryChoice(): CuprumChoice {
  return { id: 'tell-story', label: 'Tell me a story.', next: 'story' };
}

function leaveChoice(): CuprumChoice {
  return { id: 'leave', label: 'I should go.', close: true };
}

function closingChoices(): CuprumChoice[] {
  return [askAgainChoice(), tellStoryChoice(), leaveChoice()];
}

export const CUPRUM_NODES: Record<string, CuprumNode> = {
  [ROOT_NODE_ID]: {
    id: ROOT_NODE_ID,
    speech: (ctx, runtime) => {
      if (ctx.dataState === 'loading') return "The wires are still warm. Give me a moment.";
      if (ctx.dataState === 'error') return 'I lost the thread for a moment. Sit with me anyway.';
      if (ctx.dataState === 'empty') return "The board is quiet. I'm still here.";
      if (runtime.freshData) return "The wind changed since you last looked. I'm still deciding what it means.";
      if (runtime.bond >= 5) return BOND_GREETINGS[ctx.mood];
      return GREETINGS[ctx.mood];
    },
    choices: [
      { id: 'root-spark', label: "How's your spark today?", next: 'spark', effect: 'curious' },
      { id: 'root-sense', label: 'What do you sense out there?', next: 'sense', effect: 'curious' },
      { id: 'root-story', label: "Tell me something I don't know.", next: 'story', effect: 'curious' },
    ],
  },
  spark: {
    id: 'spark',
    speech: (ctx) => {
      if (ctx.dataState === 'loading') return 'Still warming up. The wires are full of half-finished thoughts.';
      if (ctx.dataState === 'error') return 'A little thin. Something in the wires went quiet.';
      if (ctx.dataState === 'empty') return "Quiet. But I'm a copper thing. We're good at waiting.";
      return SPARK_LINES[ctx.mood];
    },
    action: (ctx) => (ctx.dataState === 'ready' ? SPARK_ACTIONS[ctx.mood] : null),
    choices: [
      { id: 'spark-weight', label: "What's weighing on you?", next: 'spark-weight', effect: 'curious' },
      { id: 'spark-help', label: 'Can I help?', next: 'spark-help', effect: 'warm' },
      { id: 'spark-tease', label: "You're being dramatic.", next: 'spark-tease', effect: 'tease' },
    ],
  },
  'spark-weight': {
    id: 'spark-weight',
    speech: (ctx) => {
      if (ctx.dataState !== 'ready') return "The wires are quiet. I'm listening to the quiet.";
      const imagery = signalImagery(ctx);
      if (imagery === 'the paper weather') {
        return "The dollar keeps tapping on the window. I'm trying not to answer.";
      }
      if (ctx.riskCount === 0 && ctx.opportunityCount === 0) {
        return `I don't have a knot to show you today. Just ${imagery}, humming.`;
      }
      if (ctx.riskCount > ctx.opportunityCount) {
        return `Something in ${imagery} is leaning on the door. I can hear the hinges.`;
      }
      if (ctx.opportunityCount > ctx.riskCount) {
        return `The doors are winning today. I'm keeping one eye on the hinges.`;
      }
      return BALANCE_LINES.balanced;
    },
    action: 'Its whiskers twitch at something you cannot see.',
    choices: [
      { id: 'weight-warning', label: 'Is that a warning?', next: 'weight-warning', effect: 'curious' },
      { id: 'weight-change', label: 'What would change your mind?', next: 'weight-change', effect: 'curious' },
      askAgainChoice(),
    ],
  },
  'weight-warning': {
    id: 'weight-warning',
    speech: (ctx) => {
      const base = 'No. A weather report from a small animal. Warnings have footnotes.';
      if (ctx.confidence === 'high') return `${base} I would say it louder if I were sure.`;
      if (ctx.confidence === 'low') return `${base} And this one is still sniffing.`;
      return base;
    },
    action: 'It folds its arms, which is impressive for something with no arms.',
    choices: [
      { id: 'warning-soft', label: 'Fair enough.', next: 'warning-soft', effect: 'warm' },
      { id: 'warning-flirt', label: 'I like your weather.', next: 'warning-flirt', effect: 'warm' },
      leaveChoice(),
    ],
  },
  'warning-soft': {
    id: 'warning-soft',
    speech: "Good. You're easier to talk to than the board.",
    choices: closingChoices(),
  },
  'warning-flirt': {
    id: 'warning-flirt',
    speech: 'Careful. Compliments make me conduct better.',
    choices: closingChoices(),
  },
  'weight-change': {
    id: 'weight-change',
    speech: (ctx) => {
      if (ctx.confidence === 'low') {
        return "Ask me again after the next whisper. I don't want to guess before the wires settle.";
      }
      const imagery = signalImagery(ctx);
      return `A quieter ${imagery} would help. Or a louder one. I'm not picky. I just want it to stop pretending it isn't there.`;
    },
    choices: [
      { id: 'change-sense', label: 'What do you sense out there?', next: 'sense', effect: 'curious' },
      askAgainChoice(),
      leaveChoice(),
    ],
  },
  'spark-help': {
    id: 'spark-help',
    speech: "Stay a moment. That's already help.",
    action: 'It leans toward you, just slightly.',
    choices: [
      { id: 'help-stay', label: 'I can stay a little.', next: 'help-stay', effect: 'warm' },
      { id: 'help-joke', label: "You're easy to please.", next: 'help-joke', effect: 'tease' },
      leaveChoice(),
    ],
  },
  'help-stay': {
    id: 'help-stay',
    speech: "Good. The kettle in my chest is small, but it's enough for two.",
    choices: closingChoices(),
  },
  'help-joke': {
    id: 'help-joke',
    speech: "Easy? I'm a very old, very particular copper thing. You just happen to be good company.",
    choices: closingChoices(),
  },
  'spark-tease': {
    id: 'spark-tease',
    speech: "I'm a copper spirit. Drama is half my alloy.",
    choices: [
      { id: 'tease-more', label: 'And the other half?', next: 'tease-more', effect: 'tease' },
      { id: 'tease-soft', label: "I'm only teasing.", next: 'tease-soft', effect: 'warm' },
      leaveChoice(),
    ],
  },
  'tease-more': {
    id: 'tease-more',
    speech: "Good taste. It's the only part that doesn't tarnish.",
    choices: closingChoices(),
  },
  'tease-soft': {
    id: 'tease-soft',
    speech: "I know. I'd tarnish if you stopped.",
    choices: closingChoices(),
  },
  sense: {
    id: 'sense',
    speech: (ctx) => {
      if (ctx.dataState === 'loading') {
        return "The wires are still warm. I can hear something moving, but it hasn't decided what it is yet.";
      }
      if (ctx.dataState === 'error') return "The wires went quiet. I'm listening to the quiet.";
      if (ctx.dataState === 'empty') return "The board is cold. I'm listening anyway.";
      const imagery = signalImagery(ctx);
      const opening =
        ctx.mood === 'molten_bull'
          ? `There's a current in ${imagery}. It's warm. I'd bite it if I had more teeth.`
          : ctx.mood === 'patina_bear'
            ? `${capitalize(imagery)} is breathing slow. That's either patience or a trap.`
            : `I keep hearing ${imagery}. Not loud. Just... persistent.`;
      const balance = marketBalanceLine(ctx);
      return balance ? `${opening} ${balance}` : opening;
    },
    action: 'It closes its eyes and tilts its head, as if listening through the floor.',
    choices: [
      { id: 'sense-more', label: 'Tell me more.', next: 'sense-more', effect: 'curious' },
      { id: 'sense-worry', label: 'Should I worry?', next: 'sense-worry', effect: 'curious' },
      askAgainChoice(),
    ],
  },
  'sense-more': {
    id: 'sense-more',
    speech: (ctx) => {
      const imagery = signalImagery(ctx);
      const footnote = ctx.hasBiasWarning
        ? " There's a footnote in the air. I don't like footnotes."
        : '';
      return `I keep looking at ${imagery}. It's not the loudest thing on the board, but it's the one that hums.${footnote}`;
    },
    choices: [
      { id: 'more-worry', label: 'Should I worry?', next: 'sense-worry', effect: 'curious' },
      { id: 'more-soft', label: "You're being very mysterious.", next: 'sense-soft', effect: 'tease' },
      leaveChoice(),
    ],
  },
  'sense-worry': {
    id: 'sense-worry',
    speech: (ctx) => {
      if (ctx.dataState !== 'ready') {
        return "Worry is a big coat for a small animal. Put it on when there's weather.";
      }
      if (ctx.confidence === 'high') {
        return "I'd say it out loud: the metal is leaning, not promising. Leaning is not a promise.";
      }
      if (ctx.confidence === 'low') {
        return "I'd whisper it. The wires are still deciding, and I don't like to interrupt.";
      }
      return "I'd say it with one eye open. The metal is leaning. It hasn't picked a direction to fall.";
    },
    choices: [
      { id: 'worry-ok', label: "That's fair.", next: 'worry-ok', effect: 'warm' },
      { id: 'worry-tease', label: "You're not very reassuring.", next: 'worry-tease', effect: 'tease' },
      leaveChoice(),
    ],
  },
  'sense-soft': {
    id: 'sense-soft',
    speech: "Mystery is just weather I haven't named yet.",
    choices: closingChoices(),
  },
  'worry-ok': {
    id: 'worry-ok',
    speech: 'Fair is my favorite temperature.',
    choices: closingChoices(),
  },
  'worry-tease': {
    id: 'worry-tease',
    speech: "I'm not trying to be reassuring. I'm trying to be honest. The cute part is a bonus.",
    choices: closingChoices(),
  },
  story: {
    id: 'story',
    speech: (_ctx, runtime) =>
      CUPRUM_STORIES[runtime.storyIndex % CUPRUM_STORIES.length]?.text ??
      "That's all I remember. The rest is rust.",
    action: 'The ember in its chest dims to a listening red.',
    choices: [
      settleStoryLine(),
      { id: 'story-why', label: 'Why that one?', next: 'story-why', effect: 'curious' },
      leaveChoice(),
    ],
  },
  'story-why': {
    id: 'story-why',
    speech: (_ctx, runtime) =>
      CUPRUM_STORIES[runtime.storyIndex % CUPRUM_STORIES.length]?.reflection ??
      "Because it's mine. That's usually enough.",
    choices: [
      settleStoryLine(),
      askAgainChoice(),
      leaveChoice(),
    ],
  },
};

export const IDLE_LINE_COUNT =
  Object.values(IDLE_MOOD_LINES).reduce((total, lines) => total + lines.length, 0) +
  IDLE_GENERAL_LINES.length +
  IDLE_BOND_LINES.length +
  Object.values(IDLE_DATA_LINES).reduce((total, lines) => total + lines.length, 0);

export function createCuprumRuntime(initial: Partial<CuprumRuntime> = {}): CuprumRuntime {
  return {
    bond: initial.bond ?? 0,
    storyIndex: initial.storyIndex ?? 0,
    turn: initial.turn ?? 0,
    freshData: initial.freshData ?? false,
  };
}

export function resolveCuprumNode(
  nodeId: string,
  ctx: CuprumContext,
  runtime: CuprumRuntime,
): ResolvedCuprumNode | null {
  const node = CUPRUM_NODES[nodeId];
  if (!node) return null;
  const speech = typeof node.speech === 'function' ? node.speech(ctx, runtime) : node.speech;
  const action = typeof node.action === 'function' ? node.action(ctx, runtime) : node.action ?? null;
  return {
    id: node.id,
    speech,
    action,
    choices: node.choices,
  };
}

export function applyCuprumChoice(
  runtime: CuprumRuntime,
  choice: CuprumChoice,
): CuprumRuntime {
  const next: CuprumRuntime = {
    ...runtime,
    turn: runtime.turn + 1,
    freshData: false,
  };
  if (choice.effect) {
    next.bond = Math.min(MAX_BOND, next.bond + 1);
  }
  if (choice.advanceStory) {
    next.storyIndex = (next.storyIndex + 1) % CUPRUM_STORIES.length;
  }
  return next;
}

export function pickIdleLine(
  ctx: CuprumContext,
  runtime: CuprumRuntime,
  recentIds: readonly string[] = [],
  random: () => number = Math.random,
): CuprumIdleLine {
  if (ctx.dataState !== 'ready') {
    const dataPool = IDLE_DATA_LINES[ctx.dataState];
    return pickFromPool(dataPool, recentIds, random);
  }

  const pool: CuprumIdleLine[] = [
    ...IDLE_MOOD_LINES[ctx.mood],
    ...IDLE_GENERAL_LINES,
  ];
  if (runtime.bond >= 3) {
    pool.push(...IDLE_BOND_LINES);
  }
  return pickFromPool(pool, recentIds, random);
}

export function buildCuprumContext(
  commentary: CommentaryResponse | null,
  explain: ExplainResponse | null,
  tftAnalysis: TFTAnalysisResponse | null,
  loading = false,
  error?: string,
): CuprumContext {
  const companion = commentary?.companion_state ?? null;
  const stance = normalizeStance(
    companion?.stance ?? commentary?.ai_stance ?? tftAnalysis?.direction,
  );
  const mood = normalizeMood(companion?.mood, stance);
  const energy = normalizeEnergy(
    companion?.energy ?? riskLevelToEnergy(tftAnalysis?.risk_level),
  );
  const confidence =
    normalizeConfidence(companion?.confidence_band) ?? inferConfidence(commentary, explain);
  const topSignal =
    companion?.xai_highlights?.top_signals?.[0] ??
    explain?.variable_selection?.encoder?.[0] ??
    null;
  const sections = commentary?.sections ?? null;

  return {
    mood,
    energy,
    confidence,
    stance,
    signalCategory: topSignal?.category ?? null,
    signalLabel: topSignal?.label ?? topSignal?.feature ?? null,
    riskCount: sections?.risks?.length ?? 0,
    opportunityCount: sections?.opportunities?.length ?? 0,
    hasBiasWarning: Boolean(sections?.bias_warning),
    dataState: error
      ? 'error'
      : loading
        ? 'loading'
        : !commentary && !explain && !tftAnalysis
          ? 'empty'
          : 'ready',
    updatedAt: commentary?.generated_at ?? null,
  };
}

export function signalImagery(ctx: CuprumContext): string {
  const haystack = `${ctx.signalLabel ?? ''} ${ctx.signalCategory ?? ''}`.toLowerCase();
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
  return byCategory[ctx.signalCategory ?? ''] ?? 'the hum';
}

function pickFromPool(
  pool: readonly CuprumIdleLine[],
  recentIds: readonly string[],
  random: () => number,
): CuprumIdleLine {
  const fresh = pool.filter((line) => !recentIds.includes(line.id));
  const candidates = fresh.length > 0 ? fresh : pool;
  const index = Math.min(candidates.length - 1, Math.floor(random() * candidates.length));
  return candidates[Math.max(0, index)] ?? { id: 'quiet', text: "I'm here." };
}

function marketBalanceLine(ctx: CuprumContext): string | null {
  if (ctx.riskCount === 0 && ctx.opportunityCount === 0) return null;
  if (ctx.riskCount > ctx.opportunityCount) return BALANCE_LINES.knots;
  if (ctx.opportunityCount > ctx.riskCount) return BALANCE_LINES.doors;
  return BALANCE_LINES.balanced;
}

function normalizeMood(value: string | undefined, stance: CuprumContext['stance']): CuprumMood {
  if (value === 'molten_bull' || value === 'patina_bear' || value === 'tempered_neutral') {
    return value;
  }
  if (stance === 'BULLISH') return 'molten_bull';
  if (stance === 'BEARISH') return 'patina_bear';
  return 'tempered_neutral';
}

function normalizeEnergy(value: string | undefined): CuprumEnergy {
  if (value === 'calm' || value === 'charged' || value === 'volatile') return value;
  return 'calm';
}

function normalizeConfidence(value: string | undefined): CuprumConfidence | null {
  if (value === 'high' || value === 'medium' || value === 'low') return value;
  return null;
}

function normalizeStance(value: string | undefined): CuprumContext['stance'] {
  const normalized = (value ?? '').toUpperCase();
  if (normalized === 'BULLISH' || normalized === 'BEARISH' || normalized === 'NEUTRAL') {
    return normalized;
  }
  return 'NEUTRAL';
}

function riskLevelToEnergy(value: string | undefined): CuprumEnergy {
  if ((value ?? '').toUpperCase() === 'HIGH') return 'volatile';
  if ((value ?? '').toUpperCase() === 'MEDIUM') return 'charged';
  return 'calm';
}

function inferConfidence(
  commentary: CommentaryResponse | null,
  explain: ExplainResponse | null,
): CuprumConfidence {
  const mode = commentary?.generation_mode;
  if ((mode === 'llm' || mode === 'llm_repaired') && explain) return 'high';
  if (commentary || explain) return 'medium';
  return 'low';
}

function capitalize(value: string): string {
  return value.length > 0 ? `${value[0].toUpperCase()}${value.slice(1)}` : value;
}
