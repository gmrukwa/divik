module.exports =
/******/ (function(modules, runtime) { // webpackBootstrap
/******/ 	"use strict";
/******/ 	// The module cache
/******/ 	var installedModules = {};
/******/
/******/ 	// The require function
/******/ 	function __webpack_require__(moduleId) {
/******/
/******/ 		// Check if module is in cache
/******/ 		if(installedModules[moduleId]) {
/******/ 			return installedModules[moduleId].exports;
/******/ 		}
/******/ 		// Create a new module (and put it into the cache)
/******/ 		var module = installedModules[moduleId] = {
/******/ 			i: moduleId,
/******/ 			l: false,
/******/ 			exports: {}
/******/ 		};
/******/
/******/ 		// Execute the module function
/******/ 		modules[moduleId].call(module.exports, module, module.exports, __webpack_require__);
/******/
/******/ 		// Flag the module as loaded
/******/ 		module.l = true;
/******/
/******/ 		// Return the exports of the module
/******/ 		return module.exports;
/******/ 	}
/******/
/******/
/******/ 	__webpack_require__.ab = __dirname + "/";
/******/
/******/ 	// the startup function
/******/ 	function startup() {
/******/ 		// Load entry module and return exports
/******/ 		return __webpack_require__(34);
/******/ 	};
/******/
/******/ 	// run startup
/******/ 	return startup();
/******/ })
/************************************************************************/
/******/ ({

/***/ 6:
/***/ (function() {

eval("require")("@actions/core");


/***/ }),

/***/ 34:
/***/ (function(__unusedmodule, __unusedexports, __webpack_require__) {

// https://help.github.com/en/actions/automating-your-workflow-with-github-actions/creating-a-javascript-action
const core = __webpack_require__(6);
const fs = __webpack_require__(747);

try {
  const packageInitFile = core.getInput('packageInitFile');
  const isAlpha = core.getInput('isAlpha');
  const isBeta = core.getInput('isBeta');
  const rawVersion = core.getInput('version');

  console.log(isAlpha);
  
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

/***/ }),

/***/ 747:
/***/ (function(module) {

module.exports = require("fs");

/***/ })

/******/ });