export class TrainingApiError extends Error {
  status: number;

  constructor(message: string, status: number) {
    super(message);
    this.name = "TrainingApiError";
    this.status = status;
  }
}
