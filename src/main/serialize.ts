import { Document, documentSchema } from "../common/document";

export function deserialize(data: string): Document {
  const doc = JSON.parse(data);
  if (doc.version !== 1) {
    throw new Error("unsupported version");
  }
  return documentSchema.validateSync(doc);
}

export function serialize(doc: Document) {
  const data = JSON.stringify(doc);
  return data;
}
