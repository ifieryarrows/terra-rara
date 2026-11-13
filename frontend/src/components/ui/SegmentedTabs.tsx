import { useRef, type KeyboardEvent, type ReactNode } from 'react';
import clsx from 'clsx';
import type { LucideIcon } from 'lucide-react';

export interface SegmentedTabOption<T extends string = string> {
  value: T;
  label: string;
  badge?: ReactNode;
  icon?: LucideIcon;
  title?: string;
  disabled?: boolean;
}

export interface SegmentedTabsProps<T extends string = string> {
  value: T;
  onChange: (next: T) => void;
  options: readonly SegmentedTabOption<T>[];
  ariaLabel: string;
  size?: 'sm' | 'md';
  variant?: 'underline' | 'pill';
  onInteract?: () => void;
  className?: string;
}

/**
 * Unboxed keyboard-navigable segmented control for switching forecast bands,
 * stream focus, and horizon lenses across the open workspace canvas.
 */
export function SegmentedTabs<T extends string = string>({
  value,
  onChange,
  options,
  ariaLabel,
  size = 'md',
  variant = 'underline',
  onInteract,
  className,
}: SegmentedTabsProps<T>) {
  const buttonRefs = useRef<Array<HTMLButtonElement | null>>([]);

  const selectOption = (nextValue: T) => {
    if (nextValue !== value) {
      onInteract?.();
      onChange(nextValue);
    }
  };

  const handleKeyDown = (event: KeyboardEvent<HTMLButtonElement>, currentIndex: number) => {
    const enabledIndices = options
      .map((option, idx) => (option.disabled ? -1 : idx))
      .filter((idx) => idx >= 0);
    if (enabledIndices.length <= 1) return;

    const currentEnabledPos = enabledIndices.indexOf(currentIndex);
    let targetIndex: number | undefined;

    if (event.key === 'ArrowRight' || event.key === 'ArrowDown') {
      targetIndex = enabledIndices[(currentEnabledPos + 1) % enabledIndices.length];
    } else if (event.key === 'ArrowLeft' || event.key === 'ArrowUp') {
      targetIndex =
        enabledIndices[(currentEnabledPos - 1 + enabledIndices.length) % enabledIndices.length];
    } else if (event.key === 'Home') {
      targetIndex = enabledIndices[0];
    } else if (event.key === 'End') {
      targetIndex = enabledIndices[enabledIndices.length - 1];
    }

    if (targetIndex === undefined) return;

    event.preventDefault();
    const targetOption = options[targetIndex];
    if (!targetOption) return;
    buttonRefs.current[targetIndex]?.focus();
    selectOption(targetOption.value);
  };

  return (
    <div
      role="tablist"
      aria-label={ariaLabel}
      className={clsx(
        'cm-segmented-tabs',
        `cm-segmented-tabs--${variant}`,
        `cm-segmented-tabs--${size}`,
        className,
      )}
    >
      {options.map((option, index) => {
        const isSelected = option.value === value;
        const Icon = option.icon;

        return (
          <button
            key={option.value}
            ref={(el) => {
              buttonRefs.current[index] = el;
            }}
            type="button"
            role="tab"
            aria-selected={isSelected}
            tabIndex={isSelected ? 0 : -1}
            disabled={option.disabled}
            title={option.title}
            className={clsx(
              'cm-segmented-tab',
              isSelected && 'cm-segmented-tab--active',
            )}
            onClick={() => selectOption(option.value)}
            onKeyDown={(event) => handleKeyDown(event, index)}
          >
            {Icon ? <Icon size={size === 'sm' ? 13 : 15} aria-hidden="true" /> : null}
            <span>{option.label}</span>
            {option.badge !== undefined && option.badge !== null ? (
              <span className="cm-segmented-tab__badge cm-tabular">{option.badge}</span>
            ) : null}
          </button>
        );
      })}
    </div>
  );
}
