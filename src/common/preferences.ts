/**
 * Interim values of {@link Preferences} that is used to populate the initial state.
 */
export const STUB_PREFERENCES: Preferences = {
  constants: {
    numberOfCpus: 1,
  },
  maxCpuUsage: 1,
};

/**
 * The user preferences for the app.
 */
export interface Preferences {
  constants: {
    /** The maximum value for {@link maxCpuUsage}. */
    numberOfCpus: number;
  };
  /** The maximum number of active jobs. */
  maxCpuUsage: number;
}
