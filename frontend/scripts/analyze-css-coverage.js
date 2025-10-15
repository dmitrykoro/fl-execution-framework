/**
 * CSS Coverage Analyzer
 * Uses Playwright to navigate through the app and detect unused CSS
 */

import { chromium } from 'playwright';
import fs from 'fs';
import path from 'path';
import process from 'process';

const FRONTEND_URL = 'http://localhost:5173';
const CSS_FILE = '../frontend/src/App.css';

async function analyzeCSSCoverage() {
  console.log('🔍 Starting CSS Coverage Analysis...\n');

  const browser = await chromium.launch({ headless: true });
  const context = await browser.newContext();
  const page = await context.newPage();

  // Start CSS coverage
  await page.coverage.startCSSCoverage();

  console.log('📊 Navigating through application pages...\n');

  // Navigate through all major pages
  const routes = [
    { name: 'Dashboard', url: `${FRONTEND_URL}/` },
    { name: 'New Simulation', url: `${FRONTEND_URL}/simulations/new` },
  ];

  for (const route of routes) {
    console.log(`   Visiting: ${route.name}`);
    await page.goto(route.url, { waitUntil: 'networkidle' });
    await page.waitForTimeout(1000);

    // Expand all accordions if on New Simulation page
    if (route.name === 'New Simulation') {
      const accordionButtons = await page.locator('.accordion-button').all();
      for (const button of accordionButtons) {
        try {
          await button.click();
          await page.waitForTimeout(200);
        } catch {
          // Accordion might already be open
        }
      }
    }
  }

  // Stop CSS coverage and get results
  const coverage = await page.coverage.stopCSSCoverage();

  console.log('\n✅ Coverage collection complete!\n');

  // Analyze coverage for App.css
  const appCSSCoverage = coverage.find(entry => entry.url.includes('App.css'));

  if (!appCSSCoverage) {
    console.log('❌ Could not find App.css in coverage data');
    await browser.close();
    return;
  }

  // Calculate coverage statistics
  const totalBytes = appCSSCoverage.text.length;
  const usedBytes = appCSSCoverage.ranges.reduce((sum, range) => {
    return sum + (range.end - range.start);
  }, 0);
  const unusedBytes = totalBytes - usedBytes;
  const coveragePercent = ((usedBytes / totalBytes) * 100).toFixed(2);

  console.log('📈 Coverage Statistics:');
  console.log(`   Total CSS: ${totalBytes} bytes`);
  console.log(`   Used CSS: ${usedBytes} bytes`);
  console.log(`   Unused CSS: ${unusedBytes} bytes`);
  console.log(`   Coverage: ${coveragePercent}%\n`);

  // Extract unused CSS ranges
  const cssText = appCSSCoverage.text;
  const usedRanges = appCSSCoverage.ranges;

  // Build unused ranges
  const unusedRanges = [];
  let lastEnd = 0;

  for (const range of usedRanges) {
    if (range.start > lastEnd) {
      unusedRanges.push({ start: lastEnd, end: range.start });
    }
    lastEnd = range.end;
  }

  if (lastEnd < totalBytes) {
    unusedRanges.push({ start: lastEnd, end: totalBytes });
  }

  // Extract unused CSS snippets
  console.log('🗑️  Potentially Unused CSS Sections:\n');

  const unusedSnippets = [];
  for (const range of unusedRanges) {
    const snippet = cssText.substring(range.start, range.end).trim();
    if (snippet.length > 10) {
      // Skip tiny whitespace
      // Extract class/selector names
      const classMatches = snippet.match(/\.[a-zA-Z0-9_-]+/g) || [];
      const uniqueClasses = [...new Set(classMatches)];

      if (uniqueClasses.length > 0) {
        unusedSnippets.push({
          classes: uniqueClasses,
          snippet: snippet.substring(0, 200), // First 200 chars
          size: snippet.length,
        });
      }
    }
  }

  // Group by class name for easier review
  const classUsage = {};
  unusedSnippets.forEach(({ classes, size }) => {
    classes.forEach(cls => {
      if (!classUsage[cls]) {
        classUsage[cls] = { count: 0, totalSize: 0 };
      }
      classUsage[cls].count++;
      classUsage[cls].totalSize += size;
    });
  });

  // Sort by total size (most wasteful first)
  const sortedClasses = Object.entries(classUsage)
    .sort((a, b) => b[1].totalSize - a[1].totalSize)
    .slice(0, 20); // Top 20

  console.log('Top 20 Unused CSS Classes (by size):');
  sortedClasses.forEach(([className, stats], index) => {
    console.log(`   ${index + 1}. ${className} - ${stats.totalSize} bytes (${stats.count} rules)`);
  });

  // Write detailed report to file
  const reportPath = path.join(process.cwd(), 'css-coverage-report.json');
  const report = {
    timestamp: new Date().toISOString(),
    summary: {
      totalBytes,
      usedBytes,
      unusedBytes,
      coveragePercent: parseFloat(coveragePercent),
    },
    unusedClasses: sortedClasses.map(([className, stats]) => ({
      className,
      ...stats,
    })),
    unusedSnippets: unusedSnippets.slice(0, 50), // Top 50 snippets
  };

  fs.writeFileSync(reportPath, JSON.stringify(report, null, 2));

  console.log(`\n📄 Detailed report saved to: ${reportPath}\n`);

  await browser.close();
}

// Run the analysis
analyzeCSSCoverage().catch(console.error);
