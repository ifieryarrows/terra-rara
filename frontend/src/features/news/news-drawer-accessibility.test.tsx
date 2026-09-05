// @vitest-environment jsdom
import { useState } from 'react';
import { cleanup, render, screen, waitFor } from '@testing-library/react';
import userEvent from '@testing-library/user-event';
import '@testing-library/jest-dom/vitest';
import { afterEach, expect, it, vi } from 'vitest';
import { MotionConfig } from 'framer-motion';
import { NewsDetailDrawer } from './NewsDetailDrawer';
import type { NewsItem } from '../../types';

vi.mock('../../hooks/useNews', () => ({ useNewsDetail: () => ({ data: null }) }));
afterEach(cleanup);

it('contains keyboard focus, closes with Escape and returns focus to the selected story', async () => {
  const item = { id: 7, title: 'Sample source story', channel: 'market', url: 'https://example.com/story' } as unknown as NewsItem;
  function Harness() {
    const [open, setOpen] = useState(false);
    return <MotionConfig reducedMotion="always" transition={{ duration: 0 }}><button onClick={() => setOpen(true)}>Open source story</button><NewsDetailDrawer item={open ? item : null} onClose={() => setOpen(false)}/></MotionConfig>;
  }
  const user = userEvent.setup();
  render(<Harness/>);
  const opener = screen.getByRole('button', { name: 'Open source story' });
  await user.click(opener);
  const dialog = screen.getByRole('dialog', { name: 'News detail' });
  expect(dialog).toHaveFocus();
  expect(document.body.style.overflow).toBe('hidden');
  await user.tab({ shift: true });
  expect(dialog.contains(document.activeElement)).toBe(true);
  await user.keyboard('{Escape}');
  await waitFor(() => expect(opener).toHaveFocus());
  expect(document.body.style.overflow).toBe('');
});
