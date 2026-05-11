import { useState } from 'react';
import styles from './InputBar.module.css';

function InputBar({ disabled, categories, selectedCategory, onCategoryChange, onSubmit }) {
  const [value, setValue] = useState('');

  const submit = () => {
    const query = value.trim();
    if (!query || disabled) return;
    setValue('');
    onSubmit(query);
  };

  const handleKeyDown = (event) => {
    if (event.key === 'Enter' && !event.shiftKey) {
      event.preventDefault();
      submit();
    }
  };

  return (
    <form
      className={styles.inputArea}
      onSubmit={(event) => {
        event.preventDefault();
        submit();
      }}
    >
      <div className={styles.box}>
        <select
          className={styles.categorySelect}
          value={selectedCategory}
          disabled={disabled}
          onChange={(event) => onCategoryChange(event.target.value)}
          aria-label="查詢面向"
        >
          {categories.map((category) => (
            <option key={category.id} value={category.id}>
              {category.label}
            </option>
          ))}
        </select>
        <textarea
          value={value}
          disabled={disabled}
          onChange={(event) => setValue(event.target.value)}
          onKeyDown={handleKeyDown}
          placeholder="輸入法規問題..."
          rows={1}
        />
        <button type="submit" disabled={disabled || !value.trim()} aria-label="送出">
          ↑
        </button>
      </div>
    </form>
  );
}

export default InputBar;
