import { describe, expect, it } from 'vitest';
import {
  CUPRUM_NODES,
  ROOT_NODE_ID,
  createCuprumRuntime,
  resolveCuprumNode,
  type CuprumContext,
} from './cuprumDialogue';
import { type CuprumMood } from './cuprumVoice';

describe('Deep Dialogue Audit', () => {
  const MOODS: CuprumMood[] = ['molten_bull', 'patina_bear', 'tempered_neutral'];
  const DATA_STATES = ['ready', 'loading', 'empty', 'error'] as const;

  it('audits all nodes across moods, data states, and edge cases', () => {
    const issues: string[] = [];

    for (const mood of MOODS) {
      for (const dataState of DATA_STATES) {
        // Test both with full drivers and with empty drivers
        for (const withDrivers of [true, false]) {
          const ctx: CuprumContext = {
            mood,
            energy: 'charged',
            confidence: 'high',
            stance: mood === 'molten_bull' ? 'BULLISH' : mood === 'patina_bear' ? 'BEARISH' : 'NEUTRAL',
            dataState,
            riskCount: 1,
            opportunityCount: 1,
            signalCategory: 'Macro',
            signalLabel: 'DXY',
            hasBiasWarning: false,
            updatedAt: '2026-10-28',
            briefing: {
              driver: withDrivers
                ? {
                    phrase: 'the paper weather',
                    direction: 'warming',
                    trait: 'restless',
                    horizon: '5d',
                  }
                : null,
              secondary: withDrivers
                ? {
                    phrase: 'the far side of the world',
                    direction: 'cooling',
                    trait: 'heavy',
                    horizon: '5d',
                  }
                : null,
              drivers: withDrivers
                ? [
                    {
                      phrase: 'the paper weather',
                      label: 'Dollar Index',
                      feature: 'DXY',
                      weightPct: 31.0,
                      loudness: 'loud',
                      share: 'three in ten',
                      direction: 'warming',
                      horizon: '5d',
                    },
                    {
                      phrase: 'the far side of the world',
                      label: 'China Large-Cap',
                      feature: 'GXC',
                      weightPct: 9.3,
                      loudness: 'middling',
                      share: 'one in ten',
                      direction: 'cooling',
                      horizon: '5d',
                    },
                  ]
                : [],
              memory: { month: 'September', phrase: 'a September session' },
              memories: ['September'],
              risks: ['dollar strength pressure'],
              opportunities: ['warehouse draw support'],
              synthesis: 'copper holds steady',
              biasWarning: false,
              freshness: 'fresh',
              anomaly: false,
              horizon: '5d',
              horizonPhrase: 'the next five sessions',
            },
          };

          for (let turn = 0; turn <= 5; turn++) {
            for (let bond = 0; bond <= 9; bond += 3) {
              const runtime = createCuprumRuntime({ turn, bond });

              for (const nodeId of Object.keys(CUPRUM_NODES)) {
                const resolved = resolveCuprumNode(nodeId, ctx, runtime);
                if (!resolved) {
                  issues.push(`Node failed to resolve: ${nodeId}`);
                  continue;
                }

                const speech = resolved.speech;
                const action = resolved.action;
                const choices = resolved.choices;

                // Check 1: Empty speech
                if (!speech || speech.trim().length === 0) {
                  issues.push(`Empty speech at ${nodeId} (mood=${mood}, dataState=${dataState}, turn=${turn})`);
                }

                // Check 2: Undefined / NaN / null in speech
                if (/\b(undefined|null|NaN|\[object Object\])\b/.test(speech)) {
                  issues.push(`Leaked literal at ${nodeId}: "${speech}"`);
                }

                // Check 3: Budget overflow (> 220 chars)
                if (speech.length > 220) {
                  issues.push(`Speech budget overflow at ${nodeId} (${speech.length} chars): "${speech}"`);
                }

                // Check 4: Double periods
                if (/\.\./.test(speech) && !/\.\.\./.test(speech)) {
                  issues.push(`Double period at ${nodeId}: "${speech}"`);
                }

                // Check action
                if (action) {
                  if (/\b(undefined|null|NaN)\b/.test(action)) {
                    issues.push(`Leaked literal in action at ${nodeId}: "${action}"`);
                  }
                }

                // Check choices
                if (!choices || choices.length === 0) {
                  issues.push(`No choices at ${nodeId}`);
                } else {
                  for (const choice of choices) {
                    if (!choice.label || choice.label.trim().length === 0) {
                      issues.push(`Empty choice label at ${nodeId} (choice id=${choice.id})`);
                    }
                    if (/\b(undefined|null|NaN)\b/.test(choice.label)) {
                      issues.push(`Leaked literal in choice label at ${nodeId}: "${choice.label}"`);
                    }
                    if (!choice.close) {
                      const nextTarget = choice.next;
                      if (!nextTarget || !(nextTarget in CUPRUM_NODES || nextTarget === ROOT_NODE_ID)) {
                        issues.push(`Invalid next target '${nextTarget}' at ${nodeId} (choice id=${choice.id})`);
                      }
                    }
                  }
                }
              }
            }
          }
        }
      }
    }

    if (issues.length > 0) {
      console.error(`Audit issues found (${issues.length}):\n` + issues.slice(0, 20).join('\n'));
    }
    expect(issues).toEqual([]);
  });
});
