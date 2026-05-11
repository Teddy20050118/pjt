import CitationBlock from './CitationBlock.jsx';
import styles from './MessageBubble.module.css';

function MessageBubble({ message }) {
  const isUser = message.role === 'user';
  const lines = String(message.content || '').split('\n');

  return (
    <article className={`${styles.row} ${isUser ? styles.userRow : styles.assistantRow}`}>
      {!isUser && <div className={styles.avatar}>⌕</div>}
      <div className={`${styles.bubble} ${isUser ? styles.userBubble : styles.assistantBubble}`}>
        {lines.map((line, index) => (
          <p key={`${message.id}-${index}`}>{line || '\u00a0'}</p>
        ))}
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
