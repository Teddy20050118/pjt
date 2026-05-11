import { useEffect, useMemo, useState } from 'react';
import { streamQuery } from './api/queryClient.js';
import Sidebar from './components/Sidebar.jsx';
import MessageList from './components/MessageList.jsx';
import InputBar from './components/InputBar.jsx';
import { QUERY_CATEGORIES, getQueryCategoryLabel } from './constants/queryCategories.js';
import styles from './App.module.css';

const STORAGE_KEY = 'wastewater-law-conversations';

const initialConversations = [
  {
    id: 'sample-1',
    title: '放流水標準第五條',
    dateLabel: '今天',
    status: '已驗證',
    pinned: true,
    messages: [
      {
        id: 'm1',
        role: 'user',
        content: '我們工廠的廢水 COD 值為 120 mg/L，請問是否符合放流水標準第五條的規定？',
      },
      {
        id: 'm2',
        role: 'assistant',
        content:
          '依據《放流水標準》第五條，一般工業廢水的 COD 排放限值為 100 mg/L。\n\n您的量測值 120 mg/L 已超出標準值，建議進行後續處置。',
        citations: [
          {
            title: '引用法規',
            text: '放流水標準 §5 - 化學需氧量（COD）一般工業限值：100 mg/L（日均值）',
          },
        ],
      },
    ],
  },
  {
    id: 'sample-2',
    title: '廢水排放超標罰則',
    dateLabel: '昨天',
    status: '草稿',
    pinned: false,
    messages: [],
  },
  {
    id: 'sample-3',
    title: '申報期限與頻率確認',
    dateLabel: '05/09',
    status: '草稿',
    pinned: false,
    messages: [],
  },
  {
    id: 'sample-4',
    title: '水質檢測項目一覽',
    dateLabel: '05/08',
    status: '草稿',
    pinned: false,
    messages: [],
  },
];

function normalizeConversation(conversation) {
  return {
    pinned: false,
    ...conversation,
  };
}

function sortConversations(conversations) {
  return [...conversations].sort((first, second) => {
    if (first.pinned !== second.pinned) {
      return first.pinned ? -1 : 1;
    }
    return 0;
  });
}

function createConversation() {
  return {
    id: crypto.randomUUID(),
    title: '新增查詢',
    dateLabel: '今天',
    status: '草稿',
    pinned: false,
    messages: [],
  };
}

function getErrorMessage(error) {
  const message = String(error?.message || '');
  if (message.includes('Failed to fetch') || message.includes('NetworkError')) {
    return '無法連線到後端 API，請確認 FastAPI 服務是否已啟動。';
  }
  if (message.includes('API request failed: 4')) {
    return '查詢格式有誤，請調整問題後再試一次。';
  }
  if (message.includes('API request failed: 5')) {
    return '後端處理時發生錯誤，請稍後再試或檢查伺服器紀錄。';
  }
  return message ? `查詢失敗：${message}` : '查詢失敗，請稍後再試。';
}

function App() {
  const [conversations, setConversations] = useState(() => {
    const saved = localStorage.getItem(STORAGE_KEY);
    const parsed = saved ? JSON.parse(saved) : initialConversations;
    return parsed.map(normalizeConversation);
  });
  const [activeId, setActiveId] = useState(conversations[0]?.id);
  const [statusLabel, setStatusLabel] = useState('已驗證');
  const [isLoading, setIsLoading] = useState(false);
  const [selectedCategory, setSelectedCategory] = useState('auto');
  const [historySearch, setHistorySearch] = useState('');

  const sortedConversations = useMemo(() => sortConversations(conversations), [conversations]);
  const visibleConversations = useMemo(() => {
    const keyword = historySearch.trim().toLowerCase();
    if (!keyword) return sortedConversations;

    return sortedConversations.filter((conversation) => {
      const title = String(conversation.title || '').toLowerCase();
      const messages = (conversation.messages || [])
        .map((message) => String(message.content || '').toLowerCase())
        .join('\n');
      return title.includes(keyword) || messages.includes(keyword);
    });
  }, [historySearch, sortedConversations]);

  const activeConversation = useMemo(
    () => conversations.find((conversation) => conversation.id === activeId) || sortedConversations[0],
    [activeId, conversations, sortedConversations]
  );

  useEffect(() => {
    localStorage.setItem(STORAGE_KEY, JSON.stringify(conversations));
  }, [conversations]);

  useEffect(() => {
    if (activeConversation?.status) {
      setStatusLabel(activeConversation.status);
    }
  }, [activeConversation?.id, activeConversation?.status]);

  const updateActiveConversation = (updater) => {
    setConversations((current) =>
      current.map((conversation) =>
        conversation.id === activeConversation.id ? updater(conversation) : conversation
      )
    );
  };

  const handleNewConversation = () => {
    const next = createConversation();
    setConversations((current) => [next, ...current]);
    setActiveId(next.id);
    setStatusLabel('草稿');
  };

  const handleRenameConversation = (conversationId) => {
    const conversation = conversations.find((item) => item.id === conversationId);
    if (!conversation) return;

    const nextTitle = window.prompt('重新命名此對話', conversation.title)?.trim();
    if (!nextTitle) return;

    setConversations((current) =>
      current.map((item) =>
        item.id === conversationId ? { ...item, title: nextTitle } : item
      )
    );
  };

  const handleTogglePinConversation = (conversationId) => {
    setConversations((current) =>
      current.map((conversation) =>
        conversation.id === conversationId
          ? { ...conversation, pinned: !conversation.pinned }
          : conversation
      )
    );
  };

  const handleDeleteConversation = (conversationId) => {
    const conversation = conversations.find((item) => item.id === conversationId);
    if (!conversation) return;

    const shouldDelete = window.confirm(`刪除「${conversation.title}」這筆聊天紀錄？`);
    if (!shouldDelete) return;

    setConversations((current) => {
      const next = current.filter((item) => item.id !== conversationId);
      if (conversationId === activeConversation?.id) {
        setActiveId(sortConversations(next)[0]?.id);
      }
      return next;
    });
  };

  const handleSubmit = async (query) => {
    if (!query.trim() || isLoading) return;
    const categoryLabel = getQueryCategoryLabel(selectedCategory);

    const userMessage = {
      id: crypto.randomUUID(),
      role: 'user',
      content: query,
    };
    const assistantId = crypto.randomUUID();
    const assistantMessage = {
      id: assistantId,
      role: 'assistant',
      content: '',
      citations: [],
      categoryLabel,
      statusLabel: '解析問題中...',
      loading: true,
    };

    updateActiveConversation((conversation) => ({
      ...conversation,
      title: conversation.messages.length ? conversation.title : query.slice(0, 18),
      status: '查詢中',
      messages: [...conversation.messages, userMessage, assistantMessage],
    }));

    setIsLoading(true);
    setStatusLabel('查詢中');

    try {
      await streamQuery({
        query,
        conversationId: activeConversation.id,
        category: selectedCategory,
        onStatus: (payload) => {
          if (payload.status !== 'done') {
            const nextLabel = payload.label || '查詢中...';
            setStatusLabel(nextLabel);
            updateActiveConversation((conversation) => ({
              ...conversation,
              messages: conversation.messages.map((message) =>
                message.id === assistantId
                  ? { ...message, statusLabel: nextLabel }
                  : message
              ),
            }));
          }
        },
        onFinal: (payload) => {
          updateActiveConversation((conversation) => ({
            ...conversation,
            status: payload.waiting_for_data_input ? '需補資料' : '已驗證',
            messages: conversation.messages.map((message) =>
              message.id === assistantId
                ? {
                    ...message,
                    loading: false,
                    statusLabel: '',
                    categoryLabel,
                    content: payload.final_answer || payload.data_request_hint || '已完成查詢。',
                    citations: payload.citations || [],
                  }
                : message
            ),
          }));
          setStatusLabel(payload.waiting_for_data_input ? '需補資料' : '已驗證');
        },
        onError: (payload) => {
          throw new Error(payload.message || '查詢失敗');
        },
      });
    } catch (error) {
      const errorMessage = getErrorMessage(error);
      updateActiveConversation((conversation) => ({
        ...conversation,
        status: '錯誤',
        messages: conversation.messages.map((message) =>
          message.id === assistantId
            ? {
                ...message,
                loading: false,
                statusLabel: '',
                content: errorMessage,
                citations: [],
              }
            : message
        ),
      }));
      setStatusLabel('錯誤');
    } finally {
      setIsLoading(false);
    }
  };

  return (
    <div className={styles.shell}>
      <Sidebar
        conversations={visibleConversations}
        activeId={activeConversation?.id}
        searchQuery={historySearch}
        onSearchChange={setHistorySearch}
        onSelect={setActiveId}
        onNew={handleNewConversation}
        onRename={handleRenameConversation}
        onTogglePin={handleTogglePinConversation}
        onDelete={handleDeleteConversation}
      />
      <main className={styles.main}>
        <header className={styles.topbar}>
          <h1>{activeConversation?.title || '新增查詢'}</h1>
        </header>
        <MessageList messages={activeConversation?.messages || []} />
        <InputBar
          disabled={isLoading}
          categories={QUERY_CATEGORIES}
          selectedCategory={selectedCategory}
          onCategoryChange={setSelectedCategory}
          onSubmit={handleSubmit}
        />
      </main>
    </div>
  );
}

export default App;
