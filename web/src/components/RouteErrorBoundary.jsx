import { Component } from 'react';

/** 捕获子树渲染错误，避免整页白屏 */
export default class RouteErrorBoundary extends Component {
  constructor(props) {
    super(props);
    this.state = { error: null };
  }

  static getDerivedStateFromError(error) {
    return { error };
  }

  componentDidCatch(error, info) {
    console.error('[RouteErrorBoundary]', error, info?.componentStack);
  }

  render() {
    const { error } = this.state;
    if (error) {
      return (
        <div className="route-error-boundary" style={{ padding: 24, color: '#e8eef4' }}>
          <h2 style={{ marginTop: 0 }}>页面渲染出错</h2>
          <p style={{ color: '#8fa3b5' }}>{String(error?.message || error)}</p>
          <button
            type="button"
            className="legacy-btn primary"
            onClick={() => {
              this.setState({ error: null });
              window.location.reload();
            }}
          >
            重新加载
          </button>
        </div>
      );
    }
    return this.props.children;
  }
}
