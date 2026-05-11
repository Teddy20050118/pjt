import { useEffect, useRef } from 'react';
import MessageBubble from './MessageBubble.jsx';
import styles from './MessageList.module.css';

function MessageList({ messages }) {
  const scrollRef = useRef(null);
  const shouldStickToBottomRef = useRef(true);

  useEffect(() => {
    const node = scrollRef.current;
    if (!node || !shouldStickToBottomRef.current) return;
    node.scrollTo({ top: node.scrollHeight, behavior: 'smooth' });
  }, [messages]);

  const handleScroll = () => {
    const node = scrollRef.current;
    if (!node) return;
    const distanceFromBottom = node.scrollHeight - node.scrollTop - node.clientHeight;
    shouldStickToBottomRef.current = distanceFromBottom < 96;
  };

  return (
    <section className={styles.messages} ref={scrollRef} onScroll={handleScroll}>
      <div className={styles.inner}>
        {messages.map((message) => (
          <MessageBubble key={message.id} message={message} />
        ))}
      </div>
    </section>
  );
}

export default MessageList;
