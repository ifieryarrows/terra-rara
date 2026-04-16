export const ValidationPage = () => {
  return (
    <div className="space-y-6">
      <div className="flex items-center justify-between">
        <div>
          <h2 className="text-2xl font-bold tracking-tight text-white">Walk-Forward Validation</h2>
          <p className="text-sm text-slate-400 mt-1">Out-of-sample backtest results and baseline comparisons.</p>
        </div>
      </div>
      
      <div className="p-10 border border-slate-800 rounded-lg text-center text-slate-500">
        Validation page content will be populated via useBacktestReport query.
      </div>
    </div>
  );
};