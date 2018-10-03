local Dataset = torch.class('Dataset')




local function genNegatives(N_entities, hypernyms, method, negatives)
    local negatives = negatives
    if method == 'random' then
        negatives =  torch.rand(hypernyms:size(1), 2):mul(N_entities):ceil():cmax(1):long()
    elseif method == 'contrastive' then
        -- following Socher, randomly change one of the members of each pair, to a different entity
        local randomEntities = torch.rand(hypernyms:size(1), 1):mul(N_entities):ceil():cmax(1):long()
        local index = torch.rand(hypernyms:size(1), 1):mul(2):ceil():cmax(1):long() -- indices, between 1 and 2
        negatives = hypernyms:clone()
        negatives:scatter(2, index, randomEntities)
        m = 0
    elseif method == 'sample' then
         local s = 1
         local e = hypernyms:size(1)
         local indices = torch.randperm(negatives:size(1)):long()[{{s,e}}]
         negatives = negatives:index(1, indices)
    end
    return negatives
end

-- dataset creation
function Dataset:__init(N_entities, hypernyms, method, negatives)
    self.method = method
    self.hypernyms = hypernyms
    local N_hypernyms = hypernyms:size(1)

    self.negativesCounter = 1
    self.genNegatives = function() return genNegatives(N_entities, hypernyms, method, negatives) end

    self:regenNegatives()
    self.epoch = 0
end

function Dataset:regenNegatives()
    local negatives = self.genNegatives()
    local all_hypernyms = torch.cat(self.hypernyms, negatives, 1)
    self.hyper = all_hypernyms[{{}, 2}]
    self.hypo = all_hypernyms[{{}, 1}]
    self.target = torch.cat(torch.ones(self.hypernyms:size(1)), torch.zeros(negatives:size(1)))
end



function Dataset:size()
    return self.target:size(1)
end

function Dataset:minibatch(size)
    if not self.s or self.s + size - 1 > self:size() then
        -- new epoch; randomly shuffle dataset
        self.order = torch.randperm(self:size()):long()
        self.s = 1
        self.epoch = self.epoch + 1
        -- regenerate negatives
        self:regenNegatives()
    end

    local s = self.s
    local e = s + size - 1

    local indices = self.order[{{s,e}}]
    local hyper = self.hyper:index(1, indices)
    local hypo = self.hypo:index(1, indices)
    local target = self.target:index(1, indices)

    self.s = e + 1

    return {hyper, hypo}, target
end

function Dataset:all()
    return {self.hyper, self.hypo}, self.target
end