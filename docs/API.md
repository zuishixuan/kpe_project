## 全国路径演化

- path: /model/path/evolution
- 返回：
    - data:{List<pos_label>, List<x_axis>, List<y_axis>}
    - List<pos_label>:{每个坐标对应的技术名称}
    - List<x_axis>: {对应的年份，横坐标}
    - List<y_axis>: {对应专利所占百分比，纵坐标}


## 地区路径演化
- path: /model/path/evolution/area
- 参数
    - t_area: 地区名称 如：河南省
    - path_top: 显示几条路径
- 返回：
    - data:{List<pos_label>, List<x_axis>, List<y_axis>}
    - List<pos_label>:{每个坐标对应的技术名称}
    - List<x_axis>: {对应的年份，横坐标}
    - List<y_axis>: {对应专利所占百分比，纵坐标}


## 某一技术路径演化
- path: /model/path/evolution/id
- 参数
    - t_id: 技术id
- 返回：
    - data:{List<pos_label>, List<x_axis>, List<y_axis>}
    - List<pos_label>:{每个坐标对应名称}
    - List<x_axis>: {对应的年份，横坐标}
    - List<y_axis>: {对应专利所占百分比，纵坐标}

## 全国热点预测

- path: /model/hot/prediction
- 参数
    - t_name: 技术名称
- 返回：
    - List<x_org, y_org, x_predict, y_predict>
    - x_org\y_org:{对应统计的年份\每年相关技术专利所占百分比}
    - x_predict\y_predict: {对应统计的年份\每年相关技术专利所占百分比的预测值}

## 热点挖掘

- path: /model/hot/discovery
- 返回：
    - List<Node>
    - Node:{技术名称,该技术的研究热度值,}

## 统计

### 专利数查询

- path: /patent-query/query/patent-count
- 返回:
    - 专利数量

### 企业数查询

- path: /patent-query/query/enterprise-count
- 返回:
    - 企业数量

### 每个地区企业分布查询（省份）

- path: /patent-query/query/province-enterprise-dict
- 返回:
    - List<Node>
    - Node:{name:省份,List:<企业名>}

### 每个地区企业分布查询（城市）

- path: /patent-query/query/city-enterprise-dict
- 返回:
    - List<Node>
    - Node:{name:城市,List:<企业名>}

## 东光县企业查询

企业信息：

```json5
{
  address: "河北省沧州市东光经济开发区双高科技示范区",
  approval_data: "2020/3/26",
  business_scope: "塑料包装箱及容器制造；加工销售：流延膜、吹膜制袋，销售：塑料原料及以上产品的进出口业务。（依法须经批准的项目，经相关部门批准后方可开展经营活动）",
  business_scope_words: [
    "塑料包装箱",
    "容器",
    "流延膜",
    "吹膜制袋",
    "塑料原料"
  ],
  city: "沧州市",
  county: "东光县",
  dist: "东光经济开发区",
  establish_data: "2019/9/20",
  id: 21,
  industry: "橡胶和塑料制品业",
  name: "浩然包装有限责任公司",
  province: "河北",
  reg_capital: 5000,
  reg_status: "存续",
  research_field: null,
  town: "县区",
  year: "2019"
}
```

### 企业相关的信息统计

- path: /enp/dongguang/statistics/field
- 参数：
    - fields:必需，如无，不返回信息，主要是分号分割，可以是如下这几个信息：
        - year：根据年份统计
        - town：根据乡镇统计
        - reg_status：根据企业状态统计
        - business_scope_words：根据企业相关服务和产品信息统计； 比如：fields=year，或者fields=year;reg_status
    - area：非必需，根据相关的区域查询，乡镇信息为上述town统计得到的结果；
- 返回:
    - Map<String,List<Node>>：以统计的字段名称为key，统计列表为结果值
    - Node:{name:城市,value:相关统计数量}

### 企业相关信息分页查询

- path: /enp/dongguang/page
- 参数：
    - page：非必需，分页页码，默认1；
    - size：非必需，分页大小，默认10；
    - area：非必需，根据相关的区域查询，乡镇信息为上述town统计得到的结果；
- 返回:
    - total:总数
    - records:相关的企业信息

### 企业相关信息详情查询

- path: /enp/dongguang/info/<enp_id>
- 参数：
    - enp_id：必需，企业的id信息
- 返回：相关的企业信息

## 专利

### 企业相关信息分页查询

- path: /patent/page
- 参数：
    - page：非必需，分页页码，默认1；
    - size：非必需，分页大小，默认10；
    - applicant：非必需，也就是相关的申请者信息；
- 返回:
    - total:总数
    - records:相关的专利信息

### 企业相关信息详情查询

- path: /enp/info/<p_id>
- 参数：
    - p_id：必需，专利id信息
- 返回：相关的企业信息

### 根据年份统计专利信息

- path: /patent/statistics/year
- 参数：
    - applicant：非必需，比如applicant=华创天元实业发展有限责任公司
- 返回：
    - List<Node>
    - Node:{name:年份,value:统计值}
### 根据地区统计专利信息

- path: /patent/statistics/area
- 参数：
    - area：非必需，比如area=河北
- 返回：
    - List<Node>
    - Node:{name:地区,value:统计值}


### 根据年份统计全部企业信息

- path: /enp/statistics/year
- 返回:
    - List<Node>
    - Node:{name:年份,value:企业数}

### 根据年份统计全部企业信息，只包含河北的

- path: /enp/statistics/year_province_hebei
- 返回:
    - List<Node>
    - Node:{name:年份,value:企业数}


### 根据年份统计全部企业信息，只包含东光县的

- path: /enp/statistics/year_county_dongguang
- 返回:
    - List<Node>
    - Node:{name:年份,value:企业数}



### 东光县企业信息专利数量统计

- path: /patent-query/query/dongguang-enterprise-patent-count
- 返回:
    - List<Node>
    - Node:{name:企业,value:专利数量}

### 技术TOP4所占百分比，按年份范围和省份统计
- path: /patent/statistics/field-percent-top4-by-year-range
- 参数：
  - year_beg：必需，比如year_beg=2000
  - year_end：必需，比如year_beg=2009
  - province: 非必须，比如province=河北
- 返回:
    - List<Node>
    - Node:{name:技术关键词,value:所占百分比}
    

### 企业热词TOP_N，按省份统计
- path: /enp/statistics/words_cloud/province
- 参数：
  - top_k：必需，比如top_=10
  - province：非必需，比如province=浙江
- 返回:
    - List<Node>
    - Node:{name:技术关键词,value:所占百分比}

### 特定省份企业专利数量统计

- path: /patent/statistics/group_by_enterprise
- 参数：
  - area：非必须，比如area=河北
- 返回:
    - List<Node>
    - Node:{name:企业,value:专利数量}