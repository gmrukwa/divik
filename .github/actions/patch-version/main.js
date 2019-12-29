// https://help.github.com/en/actions/automating-your-workflow-with-github-actions/creating-a-javascript-action
const core = require('@actions/core');
const fs = require('fs');

try {
  const packageInitFile = core.getInput('packageInitFile');
  const isAlpha = core.getInput('isAlpha');
  const isBeta = core.getInput('isBeta');
  const rawVersion = core.getInput('version');
  
  var suffix = "";

  if (isAlpha) {
      suffix = "a";
  } else if (isBeta) {
      suffix = "b";
  }

  const version = rawVersion + suffix;

  var contents = fs.readFileSync(packageInitFile, 'utf8').split(/\r?\n/);
  contents[0] = "__version__ = '" + version + "'"
  contents = contents.join("");
  console.log(contents);
  fs.writeFileSync(packageInitFile, contents);

} catch (error) {
  core.setFailed(error.message);
}