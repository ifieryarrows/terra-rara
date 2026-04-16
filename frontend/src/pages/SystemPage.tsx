export const SystemPage = () => {
  return (
    <div className="space-y-6">
      <div className="flex items-center justify-between">
        <div>
          <h2 className="text-2xl font-bold tracking-tight text-white">System Status</h2>
          <p className="text-sm text-slate-400 mt-1">Infrastructure health, snapshot age, and pipeline execution logs.</p>
        </div>
      </div>
      
      <div className="p-10 border border-slate-800 rounded-lg text-center text-slate-500">
        System page content will be populated via useSystemStatus query.
      </div>
    </div>
  );
};