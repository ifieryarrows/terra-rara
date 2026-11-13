// @vitest-environment jsdom
import '@testing-library/jest-dom/vitest';
import { render } from '@testing-library/react';
import { describe, it, expect } from 'vitest';
import { VenusParticles } from './VenusParticles';

describe('VenusParticles', () => {
  it('renders canvas within an aria-hidden atmospheric container', () => {
    const { container } = render(<VenusParticles />);
    const wrapper = container.querySelector('.cm-venus-atmosphere');
    expect(wrapper).toBeInTheDocument();
    expect(wrapper).toHaveAttribute('aria-hidden', 'true');
    expect(wrapper?.querySelector('canvas')).toBeInTheDocument();
  });

  it('unmounts cleanly without throwing or memory leaks', () => {
    const { unmount } = render(<VenusParticles count={15} />);
    expect(() => unmount()).not.toThrow();
  });
});
