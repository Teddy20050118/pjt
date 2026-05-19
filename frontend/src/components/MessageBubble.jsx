import CitationBlock from './CitationBlock.jsx';
import styles from './MessageBubble.module.css';

function parseMarkdownTable(lines, startIndex) {
  const header = lines[startIndex];
  const divider = lines[startIndex + 1];
  if (!header?.includes('|') || !divider?.includes('|')) return null;

  const cellsFromLine = (line) =>
    line
      .trim()
      .replace(/^\|/, '')
      .replace(/\|$/, '')
      .split('|')
      .map((cell) => cell.trim());

  const isDivider = cellsFromLine(divider).every((cell) => /^:?-{3,}:?$/.test(cell));
  if (!isDivider) return null;

  const headers = cellsFromLine(header);
  const rows = [];
  let nextIndex = startIndex + 2;

  while (nextIndex < lines.length && lines[nextIndex].includes('|')) {
    const cells = cellsFromLine(lines[nextIndex]);
    if (cells.length !== headers.length) break;
    rows.push(cells);
    nextIndex += 1;
  }

  if (!rows.length) return null;
  return { headers, rows, nextIndex };
}

function renderMessageContent(content, messageId) {
  const lines = String(content || '').split('\n');
  const blocks = [];
  let index = 0;

  while (index < lines.length) {
    const table = parseMarkdownTable(lines, index);
    if (table) {
      blocks.push(
        <div className={styles.tableWrap} key={`${messageId}-table-${index}`}>
          <table className={styles.answerTable}>
            <thead>
              <tr>
                {table.headers.map((header, cellIndex) => (
                  <th key={`${messageId}-head-${index}-${cellIndex}`}>{header}</th>
                ))}
              </tr>
            </thead>
            <tbody>
              {table.rows.map((row, rowIndex) => (
                <tr key={`${messageId}-row-${index}-${rowIndex}`}>
                  {row.map((cell, cellIndex) => (
                    <td key={`${messageId}-cell-${index}-${rowIndex}-${cellIndex}`}>{cell}</td>
                  ))}
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      );
      index = table.nextIndex;
      continue;
    }

    blocks.push(<p key={`${messageId}-${index}`}>{lines[index] || '\u00a0'}</p>);
    index += 1;
  }

  return blocks;
}

function MessageBubble({ message }) {
  const isUser = message.role === 'user';

  return (
    <article className={`${styles.row} ${isUser ? styles.userRow : styles.assistantRow}`}>
      {!isUser && <div className={styles.avatar}>⌕</div>}
      <div className={`${styles.bubble} ${isUser ? styles.userBubble : styles.assistantBubble}`}>
        {renderMessageContent(message.content, message.id)}
        {message.categoryLabel && !isUser && (
          <div className={styles.meta}>查詢面向：{message.categoryLabel}</div>
        )}
        {message.loading && (
          <div className={styles.loading}>{message.statusLabel || '查詢中...'}</div>
        )}
        {!isUser && message.citations?.length > 0 && (
          <CitationBlock citations={message.citations} />
        )}
      </div>
      {isUser && <div className={styles.userAvatar}>我</div>}
    </article>
  );
}

export default MessageBubble;
