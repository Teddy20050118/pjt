export const QUERY_CATEGORIES = [
  {
    id: 'auto',
    label: '自動判斷',
  },
  {
    id: 'effluent_standard',
    label: '放流水標準',
  },
  {
    id: 'industry_scope',
    label: '事業/工廠類別',
  },
  {
    id: 'sewer_system',
    label: '污水下水道系統',
  },
  {
    id: 'facility_special',
    label: '建築物/總量管制',
  },
  {
    id: 'permit_plan',
    label: '許可/計畫審查',
  },
  {
    id: 'monitoring_reporting',
    label: '檢測/申報管理',
  },
  {
    id: 'penalty',
    label: '違規/裁罰',
  },
];

export function getQueryCategoryLabel(categoryId) {
  return QUERY_CATEGORIES.find((category) => category.id === categoryId)?.label || '自動判斷';
}
