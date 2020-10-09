web3.eth.defaultAccount = "0x28b36483ef3Df652d4368f082C35bFf263d1aB2c"
personal.unlockAccount(web3.eth.defaultAccount, "Qzxw@1234")
// var source = "pragma solidity ^0.4.0;contract HedgeContract1 { struct Investment { address investor; uint value; uint nowValue; uint period; uint withdrawalLimit; } address public creator; address public investAgent; address public buyAgent; uint public minimumInvestment; mapping(address => Investment) public investments;  event InvestmentMade(address accountAddress, uint amount);  modifier onlyBy(address _account) { if (msg.sender != _account) throw; _; } function HedgeContract1( uint _minimumInvestment, address _investAgent, address _buyAgent ) {  creator = msg.sender; }  function setInvestAgent(address newInvestAgent) onlyBy(creator) { investAgent = newInvestAgent; }  function setBuyAgent(address newBuyAgent) onlyBy(creator) { buyAgent = newBuyAgent; }  function setMinimumInvestment(uint newMinimumInvestment) onlyBy(creator) { minimumInvestment = newMinimumInvestment; }  function createInvestment() payable { if (msg.value < minimumInvestment) { throw; } investments[msg.sender] = Investment(msg.sender, msg.value, msg.value, 3, 1); }   function investOffer(address account, uint amount, bool invest) onlyBy(investAgent) {  if (invest) { InvestmentMade(account, amount);  } else { throw; } }  function afterInvestOffer(address account, uint amount) onlyBy(investAgent) { investments[account].nowValue = amount; }  function kill() { if (msg.sender == creator) suicide(creator); }}"
// var compiled = web3.eth.compile.solidity(source)
var abiDefinition = JSON.parse('[{"constant":true,"inputs":[],"name":"creator","outputs":[{"name":"","type":"address"}],"payable":false,"type":"function"},{"constant":true,"inputs":[],"name":"investAgent","outputs":[{"name":"","type":"address"}],"payable":false,"type":"function"},{"constant":true,"inputs":[],"name":"minimumInvestment","outputs":[{"name":"","type":"uint256"}],"payable":false,"type":"function"},{"constant":true,"inputs":[],"name":"sharpe","outputs":[{"name":"","type":"int256"}],"payable":false,"type":"function"},{"constant":false,"inputs":[],"name":"withdrawalUser","outputs":[{"name":"","type":"bool"}],"payable":false,"type":"function"},{"constant":false,"inputs":[],"name":"kill","outputs":[],"payable":false,"type":"function"},{"constant":false,"inputs":[{"name":"newMinimumInvestment","type":"uint256"}],"name":"setMinimumInvestment","outputs":[],"payable":false,"type":"function"},{"constant":true,"inputs":[],"name":"buyAgent","outputs":[{"name":"","type":"address"}],"payable":false,"type":"function"},{"constant":false,"inputs":[{"name":"sectorList","type":"uint8[]"}],"name":"createInvestment","outputs":[],"payable":true,"type":"function"},{"constant":false,"inputs":[{"name":"_returnRatio","type":"int256"},{"name":"_sharpe","type":"int256"},{"name":"_alpha","type":"int256"},{"name":"_beta","type":"int256"}],"name":"sendBuyAgent","outputs":[],"payable":true,"type":"function"},{"constant":true,"inputs":[],"name":"availableForInvestment","outputs":[{"name":"","type":"uint256"}],"payable":false,"type":"function"},{"constant":false,"inputs":[{"name":"newBuyAgent","type":"address"}],"name":"setBuyAgent","outputs":[],"payable":false,"type":"function"},{"constant":false,"inputs":[{"name":"newInvestAgent","type":"address"}],"name":"setInvestAgent","outputs":[],"payable":false,"type":"function"},{"constant":false,"inputs":[{"name":"amount","type":"uint256"},{"name":"sectorList","type":"uint8[]"}],"name":"investOffer","outputs":[],"payable":false,"type":"function"},{"constant":true,"inputs":[],"name":"beta","outputs":[{"name":"","type":"int256"}],"payable":false,"type":"function"},{"constant":false,"inputs":[],"name":"withdrawBuyAgent","outputs":[{"name":"","type":"bool"}],"payable":false,"type":"function"},{"constant":true,"inputs":[],"name":"blackListCompanies","outputs":[{"name":"","type":"uint8[48]"}],"payable":false,"type":"function"},{"constant":true,"inputs":[],"name":"originalInvestment","outputs":[{"name":"","type":"uint256"}],"payable":false,"type":"function"},{"constant":true,"inputs":[],"name":"alpha","outputs":[{"name":"","type":"int256"}],"payable":false,"type":"function"},{"constant":true,"inputs":[],"name":"returnRatio","outputs":[{"name":"","type":"int256"}],"payable":false,"type":"function"},{"constant":true,"inputs":[{"name":"investor","type":"address"}],"name":"getInvestmentCurrentValue","outputs":[{"name":"nowValue","type":"uint256"}],"payable":false,"type":"function"},{"constant":true,"inputs":[{"name":"","type":"uint256"}],"name":"investments","outputs":[{"name":"investor","type":"address"},{"name":"value","type":"uint256"},{"name":"withdrawal","type":"uint256"},{"name":"period","type":"uint256"},{"name":"withdrawalLimit","type":"uint256"}],"payable":false,"type":"function"},{"inputs":[{"name":"_minimumInvestment","type":"uint256"},{"name":"_investAgent","type":"address"},{"name":"_buyAgent","type":"address"}],"type":"constructor","payable":true},{"anonymous":false,"inputs":[{"indexed":false,"name":"amount","type":"uint256"}],"name":"InvestmentOfferByBot","type":"event"},{"anonymous":false,"inputs":[{"indexed":false,"name":"accountAddress","type":"address"},{"indexed":false,"name":"amount","type":"uint256"}],"name":"NewInvestmentByUser","type":"event"}]')
var contractC = web3.eth.contract(abiDefinition)
var hedge = contractC.at("0x543E0B5767A2274FA9E911654F9A7c16d538A9b6")
