import { useState } from 'react';
import styles from './CitationBlock.module.css';

function CitationBlock({ citations }) {
  const [expanded, setExpanded] = useState(() => new Set([0]));
  const [copiedKey, setCopiedKey] = useState('');

  const toggle = (index) => {
    setExpanded((current) => {
      const next = new Set(current);
      if (next.has(index)) {
        next.delete(index);
      } else {
        next.add(index);
      }
      return next;
    });
  };

  const copyCitation = async (citation, index) => {
    const text = [citation.title, citation.text].filter(Boolean).join('\n');
    if (!text) return;

    try {
      await navigator.clipboard.writeText(text);
      setCopiedKey(`${citation.title}-${index}`);
      window.setTimeout(() => setCopiedKey(''), 1200);
    } catch {
      setCopiedKey('');
    }
  };

  return (
    <div className={styles.citation}>
      <div className={styles.header}>引用法規</div>
      {citations.map((citation, index) => {
        const isExpanded = expanded.has(index);
        const key = `${citation.title}-${index}`;

        return (
          <div key={key} className={styles.item}>
            <button
              className={styles.titleButton}
              type="button"
              onClick={() => toggle(index)}
              aria-expanded={isExpanded}
            >
              <span>{citation.title || '引用來源'}</span>
              <span className={styles.chevron}>{isExpanded ? '收合' : '展開'}</span>
            </button>
            {isExpanded && citation.text && <p>{citation.text}</p>}
            <button
              className={styles.copyButton}
              type="button"
              onClick={() => copyCitation(citation, index)}
            >
              {copiedKey === key ? '已複製' : '複製引用'}
            </button>
          </div>
        );
      })}
    </div>
  );
}

export default CitationBlock;
