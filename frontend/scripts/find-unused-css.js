/**
 * Find Unused CSS Classes
 * Static analysis: checks which CSS classes are used in JSX files
 */

import fs from 'fs';
import path from 'path';
import { fileURLToPath } from 'url';
import { dirname } from 'path';
import process from 'process';

const __filename = fileURLToPath(import.meta.url);
const __dirname = dirname(__filename);

const CSS_FILE = path.join(__dirname, '../src/App.css');
const JSX_DIR = path.join(__dirname, '../src');

// Read CSS file and extract all class names
function extractCSSClasses(cssContent) {
  const classPattern = /\.([a-zA-Z0-9_-]+)(?=[:\s,[{>~+])/g;
  const classes = new Set();
  let match;

  while ((match = classPattern.exec(cssContent)) !== null) {
    classes.add(match[1]);
  }

  return Array.from(classes).sort();
}

// Read all JSX files recursively
function getAllJSXFiles(dir, fileList = []) {
  const files = fs.readdirSync(dir);

  files.forEach(file => {
    const filePath = path.join(dir, file);
    const stat = fs.statSync(filePath);

    if (stat.isDirectory()) {
      getAllJSXFiles(filePath, fileList);
    } else if (file.match(/\.(jsx|js|tsx|ts)$/)) {
      fileList.push(filePath);
    }
  });

  return fileList;
}

// Check if a class is used in JSX files
function findClassUsage(className, jsxFiles) {
  const usageLocations = [];

  // Patterns to match className usage
  const patterns = [
    new RegExp(`className=["']([^"']*\\b${className}\\b[^"']*)["']`, 'g'),
    new RegExp(`className=\\{[^}]*["'\`]([^"'\`]*\\b${className}\\b[^"'\`]*)["'\`][^}]*\\}`, 'g'),
  ];

  for (const file of jsxFiles) {
    const content = fs.readFileSync(file, 'utf-8');
    const relPath = path.relative(process.cwd(), file);

    for (const pattern of patterns) {
      let match;
      while ((match = pattern.exec(content)) !== null) {
        const lineNumber = content.substring(0, match.index).split('\n').length;
        usageLocations.push({
          file: relPath,
          line: lineNumber,
          context: match[0],
        });
      }
    }
  }

  return usageLocations;
}

async function analyzeCSS() {
  console.log('🔍 Analyzing CSS class usage...\n');

  // Read CSS file
  const cssContent = fs.readFileSync(CSS_FILE, 'utf-8');
  const cssClasses = extractCSSClasses(cssContent);

  console.log(`📊 Found ${cssClasses.length} CSS classes in App.css\n`);

  // Get all JSX files
  const jsxFiles = getAllJSXFiles(JSX_DIR);
  console.log(`📁 Scanning ${jsxFiles.length} JSX/JS files...\n`);

  // Analyze usage
  const unused = [];
  const used = [];
  const dynamicMaybe = [];

  // Classes that are likely dynamic or from libraries
  const knownDynamic = [
    'show',
    'active',
    'collapsed',
    'open',
    'visible',
    'hidden',
    'disabled',
    'selected',
    'checked',
    'invalid',
    'valid',
    'accordion-button',
    'accordion-body',
    'accordion-item',
    'accordion-header',
    'modal',
    'tooltip',
    'popover',
    'dropdown',
    'btn',
    'card',
    'badge',
    'alert',
    'form-control',
    'form-select',
    'nav-link',
    'nav-tabs',
    'tab-content',
    'table',
    'list-group-item',
  ];

  for (const className of cssClasses) {
    const usages = findClassUsage(className, jsxFiles);

    if (usages.length === 0) {
      // Check if it's a known dynamic class or pseudo-class variant
      if (
        knownDynamic.some(d => className.includes(d)) ||
        className.includes(':') ||
        className.includes('[')
      ) {
        dynamicMaybe.push(className);
      } else {
        unused.push(className);
      }
    } else {
      used.push({ className, usages });
    }
  }

  // Generate report
  console.log('═'.repeat(80));
  console.log('📈 USAGE SUMMARY');
  console.log('═'.repeat(80));
  console.log(`✅ Used classes: ${used.length}`);
  console.log(`⚠️  Potentially dynamic: ${dynamicMaybe.length}`);
  console.log(`❌ Unused classes: ${unused.length}\n`);

  if (unused.length > 0) {
    console.log('═'.repeat(80));
    console.log('🗑️  POTENTIALLY UNUSED CSS CLASSES');
    console.log('═'.repeat(80));
    unused.forEach((cls, index) => {
      console.log(`${index + 1}. .${cls}`);
    });
    console.log();
  }

  if (dynamicMaybe.length > 0 && dynamicMaybe.length < 30) {
    console.log('═'.repeat(80));
    console.log('⚠️  POSSIBLY DYNAMIC CLASSES (verify manually)');
    console.log('═'.repeat(80));
    dynamicMaybe.forEach((cls, index) => {
      console.log(`${index + 1}. .${cls}`);
    });
    console.log();
  }

  // Save detailed report
  const report = {
    timestamp: new Date().toISOString(),
    summary: {
      totalClasses: cssClasses.length,
      usedClasses: used.length,
      dynamicClasses: dynamicMaybe.length,
      unusedClasses: unused.length,
    },
    unused,
    dynamic: dynamicMaybe,
    used: used.map(u => ({
      className: u.className,
      usageCount: u.usages.length,
      locations: u.usages,
    })),
  };

  const reportPath = path.join(process.cwd(), 'css-usage-report.json');
  fs.writeFileSync(reportPath, JSON.stringify(report, null, 2));

  console.log(`📄 Detailed report saved to: ${reportPath}`);
  console.log(`\n✨ Analysis complete!\n`);
}

analyzeCSS().catch(console.error);
