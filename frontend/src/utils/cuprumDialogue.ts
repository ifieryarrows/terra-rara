import type {
  CommentaryResponse,
  ExplainResponse,
  TFTAnalysisResponse,
} from '../types';
import {
  buildCuprumBriefing,
  composeCuprumReply,
  type CuprumConfidence,
  type CuprumDataState,
  type CuprumEnergy,
  type CuprumMood,
  type CuprumVoiceContext,
} from './cuprumVoice';

export type {
  CuprumConfidence,
  CuprumDataState,
  CuprumEnergy,
  CuprumMood,
} from './cuprumVoice';

export interface CuprumContext extends CuprumVoiceContext {
  stance: 'BULLISH' | 'BEARISH' | 'NEUTRAL';
  signalCategory: string | null;
  signalLabel: string | null;
  /** Kept on the dialogue context for callers; the voice reads briefing.biasWarning. */
  hasBiasWarning: boolean;
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
  choices:
    | CuprumChoice[]
    | ((ctx: CuprumContext, runtime: CuprumRuntime) => CuprumChoice[]);
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

const SPARK_ACTIONS: Record<CuprumMood, string> = {
  molten_bull: 'The ember in its chest brightens.',
  patina_bear: 'A little green flame curls around its ear.',
  tempered_neutral: 'It taps one claw against the glass, very softly.',
};

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
      if (runtime.freshData && runtime.turn === 0) {
        return "The wind changed since you last looked. I'm still deciding what it means.";
      }
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
    speech: (ctx, runtime) => composeCuprumReply('spark', ctx, runtime),
    action: (ctx) => (ctx.dataState === 'ready' ? SPARK_ACTIONS[ctx.mood] : null),
    choices: [
      { id: 'spark-weight', label: "What's weighing on you?", next: 'spark-weight', effect: 'curious' },
      { id: 'spark-help', label: 'Can I help?', next: 'spark-help', effect: 'warm' },
      { id: 'spark-tease', label: "You're being dramatic.", next: 'spark-tease', effect: 'tease' },
    ],
  },
  'spark-weight': {
    id: 'spark-weight',
    speech: (ctx, runtime) => composeCuprumReply('weight', ctx, runtime),
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
    speech: (ctx, runtime) => composeCuprumReply('weight-change', ctx, runtime),
    choices: [
      { id: 'change-now', label: 'What changed?', next: 'change', effect: 'curious' },
      { id: 'change-sense', label: 'What do you sense out there?', next: 'sense', effect: 'curious' },
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
    speech: (ctx, runtime) => composeCuprumReply('sense', ctx, runtime),
    action: 'It closes its eyes and tilts its head, as if listening through the floor.',
    choices: (ctx) => [
      { id: 'sense-more', label: 'Tell me more.', next: 'sense-more', effect: 'curious' },
      { id: 'sense-worry', label: 'Should I worry?', next: 'sense-worry', effect: 'curious' },
      ctx.dataState === 'ready' && ctx.briefing.driver
        ? { id: 'sense-driver', label: "What's the loose thread?", next: 'driver-detail', effect: 'curious' }
        : askAgainChoice(),
    ],
  },
  'sense-more': {
    id: 'sense-more',
    speech: (ctx, runtime) => composeCuprumReply('sense-more', ctx, runtime),
    choices: (ctx) => [
      ctx.dataState !== 'ready'
        ? askAgainChoice()
        : ctx.briefing.memory
          ? { id: 'more-memory', label: 'Why that old day?', next: 'memory-why', effect: 'curious' }
          : { id: 'more-driver', label: "What's the loose thread?", next: 'driver-detail', effect: 'curious' },
      { id: 'more-worry', label: 'Should I worry?', next: 'sense-worry', effect: 'curious' },
      leaveChoice(),
    ],
  },
  'sense-worry': {
    id: 'sense-worry',
    speech: (ctx, runtime) => composeCuprumReply('sense-worry', ctx, runtime),
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
  'driver-detail': {
    id: 'driver-detail',
    speech: (ctx, runtime) => composeCuprumReply('driver-detail', ctx, runtime),
    action: 'Its tail flicks toward a corner of the board that is not there.',
    choices: [
      { id: 'driver-risk', label: 'What worries you?', next: 'risk', effect: 'curious' },
      { id: 'driver-door', label: "What's opening up?", next: 'opportunity', effect: 'curious' },
      leaveChoice(),
    ],
  },
  change: {
    id: 'change',
    speech: (ctx, runtime) => composeCuprumReply('change', ctx, runtime),
    action: 'It turns one ear toward the window, then the other.',
    choices: [
      { id: 'change-sense', label: 'What do you sense out there?', next: 'sense', effect: 'curious' },
      askAgainChoice(),
      leaveChoice(),
    ],
  },
  'memory-why': {
    id: 'memory-why',
    speech: (ctx, runtime) => composeCuprumReply('memory-why', ctx, runtime),
    action: 'A thin green light moves under its patina.',
    choices: [
      { id: 'memory-driver', label: "What's the loudest thing?", next: 'driver-detail', effect: 'curious' },
      { id: 'memory-soft', label: "You're being very mysterious.", next: 'sense-soft', effect: 'tease' },
      leaveChoice(),
    ],
  },
  risk: {
    id: 'risk',
    speech: (ctx, runtime) => composeCuprumReply('risk', ctx, runtime),
    action: 'It taps one claw on the glass, once.',
    choices: [
      { id: 'risk-change', label: 'What changed?', next: 'change', effect: 'curious' },
      { id: 'risk-worry', label: 'Should I worry?', next: 'sense-worry', effect: 'curious' },
      leaveChoice(),
    ],
  },
  opportunity: {
    id: 'opportunity',
    speech: (ctx, runtime) => composeCuprumReply('opportunity', ctx, runtime),
    action: 'A small ember brightens behind its ribs.',
    choices: [
      { id: 'door-change', label: 'What changed?', next: 'change', effect: 'curious' },
      { id: 'door-worry', label: 'Should I worry?', next: 'sense-worry', effect: 'curious' },
      leaveChoice(),
    ],
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
  const choices = typeof node.choices === 'function' ? node.choices(ctx, runtime) : node.choices;
  return {
    id: node.id,
    speech,
    action,
    choices,
  };
}

export function applyCuprumChoice(
  runtime: CuprumRuntime,
  choice: CuprumChoice,
): CuprumRuntime {
  const next: CuprumRuntime = {
    ...runtime,
    turn: runtime.turn + 1,
    // freshData stays sticky for the whole conversation: the root greeting and
    // the "what changed" node must both see the same fresh snapshot. It is reset
    // per conversation when the panel opens the next one.
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
  const briefing = buildCuprumBriefing(commentary, explain, tftAnalysis);

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
    briefing,
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
