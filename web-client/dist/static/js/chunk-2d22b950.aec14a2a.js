(window["webpackJsonp"]=window["webpackJsonp"]||[]).push([["chunk-2d22b950"],{f01a:function(t,e,a){"use strict";a.r(e);var n=function(){var t=this,e=t.$createElement,a=t._self._c||e;return a("el-card",{staticClass:"box-card",attrs:{shadow:"hover"}},[a("el-table",{staticStyle:{width:"100%"},attrs:{data:t.tableData,border:"",height:"650"}},[a("el-table-column",{attrs:{fixed:"",prop:"id",label:"id",width:"300"}}),a("el-table-column",{attrs:{prop:"word",label:"关键词",width:"300"}}),a("el-table-column",{attrs:{"show-password":"",prop:"main_word",label:"主要关键词",width:"350"}}),a("el-table-column",{attrs:{prop:"status",label:"状态",width:"300"}}),a("el-table-column",{attrs:{label:"操作"},scopedSlots:t._u([{key:"default",fn:function(e){return[a("el-button",{attrs:{size:"mini"},on:{click:function(a){return t.handleEdit(e.$index,e.row)}}},[t._v("编辑")]),a("el-button",{attrs:{size:"mini",type:"danger"},on:{click:function(a){return t.handleDelete(e.$index,e.row)}}},[t._v("删除")])]}}])})],1),a("div",{staticClass:"block",staticStyle:{float:"right"}},[a("el-pagination",{attrs:{background:"","current-page":this.page,"page-sizes":[10,50,100,200],"page-size":this.offset,layout:"total, sizes, prev, pager, next, jumper",total:this.totalNum},on:{"size-change":t.handleSizeChange,"current-change":t.handleCurrentChange}})],1)],1)},l=[],o=a("b775");function i(t){return Object(o["a"])({url:"/synonym-dict/page",method:"get",params:t})}var r={data:function(){return{page:1,offset:10,totalNum:1e3,tableData:[]}},created:function(){this.filter()},methods:{handleClick:function(t){console.log(t)},filter:function(){var t=this,e={pageno:this.page,pagesize:this.offset};i(e).then((function(e){var a=t.$createElement;t.tableData=e.data.data,console.log(t.tableData),t.totalNum=e.data["totalnum"],t.$notify({title:"搜索成功",message:a("i",{style:"color: teal"},"搜索成功"),duration:1e3})})).catch((function(t){console.log(t)}))},handleSizeChange:function(t){this.offset=t,this.filter()},handleCurrentChange:function(t){this.page=t,this.filter()},handleEdit:function(t,e){console.log(t,e)},handleDelete:function(t,e){console.log(t,e)}}},s=r,c=a("cba8"),u=Object(c["a"])(s,n,l,!1,null,"379b6f12",null);e["default"]=u.exports}}]);