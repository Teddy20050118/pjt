import { useState } from 'react';
import styles from './Sidebar.module.css';

function Sidebar({
  conversations,
  activeId,
  searchQuery,
  onSearchChange,
  onSelect,
  onNew,
  onRename,
  onTogglePin,
  onDelete,
}) {
  const [openMenuId, setOpenMenuId] = useState(null);

  const handleSelect = (conversationId) => {
    setOpenMenuId(null);
    onSelect(conversationId);
  };

  const handleMenuAction = (action, conversationId) => {
    setOpenMenuId(null);
    action(conversationId);
  };

  return (
    <aside className={styles.sidebar}>
      <div className={styles.header}>
        <div className={styles.title}>查詢記錄</div>
        <div className={styles.subtitle}>最近 7 天</div>
        <input
          className={styles.searchInput}
          value={searchQuery}
          onChange={(event) => onSearchChange(event.target.value)}
          placeholder="搜尋聊天..."
          aria-label="搜尋聊天記錄"
        />
      </div>

      <nav className={styles.list}>
        {conversations.length === 0 && (
          <div className={styles.empty}>找不到符合的聊天</div>
        )}
        {conversations.map((conversation) => (
          <div
            key={conversation.id}
            className={`${styles.item} ${conversation.id === activeId ? styles.active : ''}`}
          >
            <button
              className={styles.itemMain}
              onClick={() => handleSelect(conversation.id)}
              type="button"
            >
              <span className={styles.itemTitle}>
                {conversation.pinned && <span className={styles.pin}>釘選</span>}
                {conversation.title}
              </span>
              <small>{conversation.dateLabel}</small>
            </button>

            <button
              className={styles.menuButton}
              aria-label={`管理 ${conversation.title}`}
              aria-expanded={openMenuId === conversation.id}
              onClick={() =>
                setOpenMenuId((current) =>
                  current === conversation.id ? null : conversation.id
                )
              }
              type="button"
            >
              ⋯
            </button>

            {openMenuId === conversation.id && (
              <div className={styles.menu}>
                <button
                  type="button"
                  onClick={() => handleMenuAction(onTogglePin, conversation.id)}
                >
                  {conversation.pinned ? '取消釘選' : '釘選'}
                </button>
                <button
                  type="button"
                  onClick={() => handleMenuAction(onRename, conversation.id)}
                >
                  重新命名
                </button>
                <button
                  className={styles.danger}
                  type="button"
                  onClick={() => handleMenuAction(onDelete, conversation.id)}
                >
                  刪除紀錄
                </button>
              </div>
            )}
          </div>
        ))}
      </nav>

      <div className={styles.footer}>
        <button className={styles.newButton} onClick={onNew} type="button">
          <span>＋</span>
          新增查詢
        </button>
      </div>
    </aside>
  );
}

export default Sidebar;
