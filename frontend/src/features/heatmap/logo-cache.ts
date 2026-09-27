const failedLogos = new Set<string>();
const loadedLogos = new Set<string>();

export const hasFailedLogo = (url: string) => failedLogos.has(url);
export const markLogoFailed = (url: string) => failedLogos.add(url);
export const hasLoadedLogo = (url: string) => loadedLogos.has(url);
export const markLogoLoaded = (url: string) => loadedLogos.add(url);
export const resetLogoCacheForTests = () => {
  failedLogos.clear();
  loadedLogos.clear();
};
