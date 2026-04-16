import { BrowserRouter, Routes, Route } from 'react-router-dom';
import { QueryClient, QueryClientProvider } from '@tanstack/react-query';

import { AppShell } from './layouts/AppShell';
import { OverviewPage } from './pages/OverviewPage';
import { ModelsPage } from './pages/ModelsPage';
import { ValidationPage } from './pages/ValidationPage';
import { SystemPage } from './pages/SystemPage';

import './App.css';

const queryClient = new QueryClient({
  defaultOptions: {
    queries: {
      refetchOnWindowFocus: false,
      retry: 1,
    },
  },
});

function App() {
  return (
    <QueryClientProvider client={queryClient}>
      <BrowserRouter>
        <Routes>
          <Route path="/" element={<AppShell />}>
            <Route index element={<OverviewPage />} />
            <Route path="models" element={<ModelsPage />} />
            <Route path="validation" element={<ValidationPage />} />
            <Route path="system" element={<SystemPage />} />
          </Route>
        </Routes>
      </BrowserRouter>
    </QueryClientProvider>
  );
}

export default App;
