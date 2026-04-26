"use client";

interface OperationStubProps {
  title: string;
  description: string;
  backendNote: string;
  status?: string;
}

export function OperationStub({ title, description, backendNote, status }: OperationStubProps) {
  return (
    <div className="card-editorial p-6 max-w-3xl">
      <h2 className="font-display text-display-md font-bold text-burgundy mb-2">{title}</h2>
      <p className="font-body text-body-sm text-foreground mb-4">{description}</p>
      <div className="font-sans text-caption uppercase tracking-wider text-muted-foreground mb-1">
        Backend
      </div>
      <p className="font-body text-body-sm text-foreground mb-5">{backendNote}</p>
      <div className="border-t border-parchment pt-3 font-sans text-caption text-muted-foreground">
        {status ?? "Stub — implementation pending. Operation logic will land in /lib/operations and the matching component."}
      </div>
    </div>
  );
}
