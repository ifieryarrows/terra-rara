export const ModelsPage = () => {
  return (
    <div className="space-y-6">
      <div className="flex items-center justify-between">
        <div>
          <h2 className="text-2xl font-bold tracking-tight text-white">TFT-ASRO Model</h2>
          <p className="text-sm text-slate-400 mt-1">Deep Learning configuration, metrics, and variable importance.</p>
        </div>
      </div>
      
      <div className="p-10 border border-slate-800 rounded-lg text-center text-slate-500">
        Models page content will be populated via useTftModelSummary query.
      </div>
    </div>
  );
};