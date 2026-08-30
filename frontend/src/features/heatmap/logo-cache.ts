const failedLogos = new Set<string>();

export const hasFailedLogo = (url: string) => failedLogos.has(url);
export const markLogoFailed = (url: string) => failedLogos.add(url);
export const resetFailedLogosForTests = () => failedLogos.clear();
