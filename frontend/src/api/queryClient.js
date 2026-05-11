export async function streamQuery({ query, conversationId, category = 'auto', onStatus, onFinal, onError }) {
  const response = await fetch('/api/query', {
    method: 'POST',
    headers: {
      'Content-Type': 'application/json',
      Accept: 'text/event-stream',
    },
    body: JSON.stringify({
      query,
      conversation_id: conversationId,
      category,
    }),
  });

  if (!response.ok || !response.body) {
    throw new Error(`API request failed: ${response.status}`);
  }

  const reader = response.body.getReader();
  const decoder = new TextDecoder();
  let buffer = '';

  const dispatchEvent = (rawEvent) => {
    const lines = rawEvent.split('\n');
    const eventLine = lines.find((line) => line.startsWith('event:'));
    const dataLine = lines.find((line) => line.startsWith('data:'));
    if (!eventLine || !dataLine) return;

    const event = eventLine.replace('event:', '').trim();
    const payload = JSON.parse(dataLine.replace('data:', '').trim());

    if (event === 'status') onStatus?.(payload);
    if (event === 'final') onFinal?.(payload);
    if (event === 'error') onError?.(payload);
  };

  while (true) {
    const { value, done } = await reader.read();
    if (done) break;

    buffer += decoder.decode(value, { stream: true });
    const events = buffer.split('\n\n');
    buffer = events.pop() || '';
    events.forEach(dispatchEvent);
  }

  if (buffer.trim()) {
    dispatchEvent(buffer);
  }
}
