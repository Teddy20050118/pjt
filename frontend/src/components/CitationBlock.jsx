import { useState } from 'react';
import styles from './CitationBlock.module.css';

const BARE_ARTICLE_PATTERN = /^\s*第\s*([0-9一二三四五六七八九十百]+)\s*條\s*$/;
const ARTICLE_PATTERN = /第\s*([0-9一二三四五六七八九十百]+)\s*條/;
const BARE_TABLE_PATTERN = /^\s*附表\s*([0-9一二三四五六七八九十百]+)\s*$/;
const TABLE_PATTERN = /附表\s*([0-9一二三四五六七八九十百]+)/;

function normalizeTitle(title = '') {
  return String(title).replace(/\s+/g, ' ').trim();
}

function isBareArticleTitle(title) {
  return BARE_ARTICLE_PATTERN.test(normalizeTitle(title));
}

function isBareTableTitle(title) {
  return BARE_TABLE_PATTERN.test(normalizeTitle(title));
}

function extractArticleNumber(title) {
  return normalizeTitle(title).match(ARTICLE_PATTERN)?.[1] || '';
}

function extractTableNumber(title) {
  return normalizeTitle(title).match(TABLE_PATTERN)?.[1] || '';
}

function titleWithoutArticle(title) {
  return normalizeTitle(title).replace(ARTICLE_PATTERN, '').trim();
}

function canonicalizeCitation(citation) {
  const title = normalizeTitle(citation.title);
  const lawName = normalizeTitle(citation.law_name);
  const articleNo = extractArticleNumber(title) || extractArticleNumber(citation.article || '');
  const tableNo = extractTableNumber(title) || extractTableNumber(citation.source_file || '');

  if (articleNo) {
    const hasLawName = Boolean(lawName) || !isBareArticleTitle(title);
    const keyLaw = lawName || (isBareArticleTitle(title) ? '__unknown_law__' : titleWithoutArticle(title));
    return {
      key: ['article', keyLaw, articleNo].join('|'),
      specificity: hasLawName ? 30 : 10,
    };
  }

  if (tableNo) {
    return {
      key: ['table', '', tableNo].join('|'),
      specificity: isBareTableTitle(title) ? 10 : 30,
    };
  }

  return {
    key: ['law', lawName || title, ''].join('|'),
    specificity: 20,
  };
}

function mergeCitationFields(primary, secondary) {
  return {
    ...primary,
    text: primary.text || secondary.text,
    source_file: primary.source_file || secondary.source_file,
    law_name: primary.law_name || secondary.law_name,
    article: primary.article || secondary.article,
    score: primary.score || secondary.score,
    type: primary.type || secondary.type,
  };
}

function cleanCitations(citations = []) {
  const groups = new Map();
  const order = [];

  citations
    .map((citation) => ({
      ...citation,
      title: normalizeTitle(citation.title),
    }))
    .filter((citation) => citation.title)
    .forEach((citation) => {
      const canonical = canonicalizeCitation(citation);
      const existing = groups.get(canonical.key);

      if (!existing) {
        groups.set(canonical.key, citation);
        order.push(canonical.key);
        return;
      }

      const existingCanonical = canonicalizeCitation(existing);
      if (canonical.specificity > existingCanonical.specificity) {
        groups.set(canonical.key, mergeCitationFields(citation, existing));
      } else {
        groups.set(canonical.key, mergeCitationFields(existing, citation));
      }
    });

  absorbBareReferences(groups);
  return order.map((key) => groups.get(key)).filter(Boolean);
}

function absorbBareReferences(groups) {
  const bareEntries = Array.from(groups.entries()).filter(([, citation]) => {
    const canonical = canonicalizeCitation(citation);
    return (
      (canonical.key.startsWith('article|') && isBareArticleTitle(citation.title)) ||
      (canonical.key.startsWith('table|') && isBareTableTitle(citation.title))
    );
  });

  bareEntries.forEach(([bareKey, bareCitation]) => {
    const bareCanonical = canonicalizeCitation(bareCitation);
    const matches = Array.from(groups.entries()).filter(([key, candidate]) => {
      if (key === bareKey) return false;
      const candidateCanonical = canonicalizeCitation(candidate);
      const sameArticle =
        extractArticleNumber(candidate.title || candidate.article || '') ===
        extractArticleNumber(bareCitation.title || bareCitation.article || '');
      const sameTable =
        extractTableNumber(candidate.title || candidate.source_file || '') ===
        extractTableNumber(bareCitation.title || bareCitation.source_file || '');

      return (
        candidateCanonical.specificity > bareCanonical.specificity &&
        ((bareCanonical.key.startsWith('article|') && sameArticle) ||
          (bareCanonical.key.startsWith('table|') && sameTable))
      );
    });

    if (matches.length === 1) {
      const [targetKey, targetCitation] = matches[0];
      groups.set(targetKey, mergeCitationFields(targetCitation, bareCitation));
      groups.delete(bareKey);
    } else if (matches.length > 1) {
      groups.delete(bareKey);
    }
  });
}

function CitationBlock({ citations }) {
  const cleanedCitations = cleanCitations(citations);
  const [expanded, setExpanded] = useState(() => new Set([0]));
  const [copiedKey, setCopiedKey] = useState('');

  if (!cleanedCitations.length) return null;

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
      {cleanedCitations.map((citation, index) => {
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
