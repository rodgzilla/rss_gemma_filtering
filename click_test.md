# Click Test Note

Embed the test chart below, then click a green point. The clicked link should appear underneath.

<iframe src="click_test.html" style="width:100%;height:400px;border:none;"></iframe>

```dataviewjs
const container = this.container;
const el = container.createEl('div', {
    attr: { style: 'padding: 8px 0; font-size: 13px; min-height: 24px;' }
});
el.setText('No article selected yet — click a point in the chart above.');

window.addEventListener('message', (event) => {
    if (!event.data || event.data.type !== 'rss-viz-click') return;
    const { url, title } = event.data;
    el.empty();
    el.createEl('span', { text: '→ ', attr: { style: 'color: #a6adc8;' } });
    el.createEl('a', { text: title || url, href: url, attr: { style: 'color: #89b4fa;' } });
});
```
